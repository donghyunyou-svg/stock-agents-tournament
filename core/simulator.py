"""
10개 에이전트를 하루 단위로 실행하고, 주간 단위로 스코어링/전략공유(진화)를 수행하는
오케스트레이터.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List

import pandas as pd

from .agents import BaseAgent, build_default_agents
from .data import MarketDataProvider, Instrument
from .evolution import share_and_evolve
from .portfolio import Order, VirtualPortfolio
from .scoring import DailyResult, WeeklyTournament


class TournamentSimulator:
    # --- 모든 에이전트에 공통으로 적용되는 리스크 관리(서킷브레이커) 설정 ---
    # 전략과 무관하게, 어떤 에이전트든 고점 대비 이 이상 손실이 나면 강제로 전량 현금화하고
    # 일정 기간 매수를 정지시켜 손실을 제한한다 ("성과를 극대화"하려면 하락장에서 살아남는 것이
    # 가장 중요하다는 원칙).
    MAX_DRAWDOWN = 0.15       # 고점 대비 15% 손실 시 서킷브레이커 발동
    COOLDOWN_DAYS = 3         # 발동 후 매수 정지 기간(거래일)

    def __init__(self, provider: MarketDataProvider, budget_per_agent: float = 10_000_000,
                 agents: List[BaseAgent] = None, history_lookback_days: int = 400):
        self.provider = provider
        self.universe = provider.get_universe()
        self.tickers = [i.ticker for i in self.universe]
        self.agents = agents or build_default_agents()
        self.portfolios: Dict[str, VirtualPortfolio] = {
            a.agent_id: VirtualPortfolio(a.agent_id, budget_per_agent) for a in self.agents
        }
        self.tournament = WeeklyTournament()
        self.history_lookback_days = history_lookback_days
        self.evolution_log: List[Dict] = []
        self._week_start_nav: Dict[str, float] = {}
        self._current_iso_week = None

    def _fetch_history(self, end_day: date) -> Dict[str, pd.DataFrame]:
        start = end_day - timedelta(days=self.history_lookback_days * 2)  # 영업일 감안 여유
        out = {}
        for t in self.tickers:
            try:
                df = self.provider.get_history(t, start, end_day)
            except Exception as e:  # 실거래 API 호출 실패 시 해당 종목만 건너뛰고 계속 진행
                print(f"[경고] {t} 시세 조회 실패, 이 종목은 오늘 건너뜁니다: {e}", file=sys.stderr)
                continue
            if df is not None and not df.empty:
                out[t] = df
        return out

    def run_day(self, day: date):
        history = self._fetch_history(day)
        if not history:
            return None
        prices = {t: df["Close"].iloc[-1] for t, df in history.items() if not df.empty}
        macro = dict(self.provider.get_macro_snapshot(day))

        # 종목별 재무지표(PER/PBR 등)를 조회해 macro 딕셔너리에 실어 에이전트에 전달한다
        # (실패해도 해당 종목만 빠지고 전체 사이클은 계속 진행).
        fundamentals: Dict[str, Dict[str, float]] = {}
        for t in history.keys():
            try:
                f = self.provider.get_fundamentals(t)
            except Exception as e:
                print(f"[경고] {t} 재무지표 조회 실패: {e}", file=sys.stderr)
                continue
            if f:
                fundamentals[t] = f
        macro["fundamentals"] = fundamentals

        iso_week = day.isocalendar()[:2]  # (year, week)
        if self._current_iso_week is None:
            self._current_iso_week = iso_week
            self._week_start_nav = {
                a.agent_id: self.portfolios[a.agent_id].cash for a in self.agents
            }
        elif iso_week != self._current_iso_week:
            self._close_week()
            self._current_iso_week = iso_week
            self._week_start_nav = {
                a.agent_id: self.portfolios[a.agent_id].history[-1]["nav"]
                if self.portfolios[a.agent_id].history else self.portfolios[a.agent_id].cash
                for a in self.agents
            }

        for agent in self.agents:
            pf = self.portfolios[agent.agent_id]

            if pf.halt_days_remaining > 0:
                # 서킷브레이커 발동 중: 신규 매수 없이 현금 보유만 유지, 잔여일 차감
                orders = [Order(t, "SELL", q, "circuit-breaker-cooldown")
                          for t, q in pf.holdings.items()]
                pf.halt_days_remaining -= 1
            elif pf.drawdown_from_peak(prices) >= self.MAX_DRAWDOWN:
                # 고점 대비 손실이 한도를 넘으면 전량 현금화하고 냉각 기간에 진입
                orders = [Order(t, "SELL", q, "circuit-breaker-trigger")
                          for t, q in pf.holdings.items()]
                pf.halt_days_remaining = self.COOLDOWN_DAYS
            else:
                orders = agent.decide(day, history, macro, self.universe, pf, prices)

            pf.execute(orders, prices)
            pf.mark_to_market(day.isoformat(), prices)

        daily_returns = {a.agent_id: self.portfolios[a.agent_id].daily_return() for a in self.agents}
        return self.tournament.record_day(day.isoformat(), daily_returns)

    def _close_week(self):
        weekly_returns = {}
        for a in self.agents:
            pf = self.portfolios[a.agent_id]
            start_nav = self._week_start_nav.get(a.agent_id, pf.budget)
            end_nav = pf.history[-1]["nav"] if pf.history else pf.budget
            weekly_returns[a.agent_id] = (end_nav / start_nav - 1) if start_nav > 0 else 0.0
        week_label = f"{self._current_iso_week[0]}-W{self._current_iso_week[1]:02d}"
        log = share_and_evolve(self.agents, weekly_returns, week_label)
        self.evolution_log.append(log)
        self.tournament.reset_week()

    def run(self, trading_days: List[date]):
        for d in trading_days:
            self.run_day(d)
        # 마지막 주는 아직 마감되지 않았을 수 있으므로 강제 마감(리포트용)
        return self.summary()

    def summary(self) -> Dict:
        leaderboard = self.tournament.leaderboard()
        nav_summary = {
            a.agent_id: {
                "name": a.name,
                "final_nav": self.portfolios[a.agent_id].history[-1]["nav"]
                if self.portfolios[a.agent_id].history else self.portfolios[a.agent_id].budget,
                "total_return_pct": round(self.portfolios[a.agent_id].total_return() * 100, 2),
                "params": dict(a.params),
            }
            for a in self.agents
        }
        return {
            "leaderboard": leaderboard,
            "nav_summary": nav_summary,
            "evolution_log": self.evolution_log,
        }

    def save_state(self, path: str):
        state = {
            "portfolios": {aid: pf.to_dict() for aid, pf in self.portfolios.items()},
            "tournament": self.tournament.to_dict(),
            "evolution_log": self.evolution_log,
            "agent_params": {a.agent_id: a.params for a in self.agents},
            "current_iso_week": list(self._current_iso_week) if self._current_iso_week else None,
            "week_start_nav": self._week_start_nav,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, default=str)

    def load_state(self, path: str):
        """이전 실행(GitHub Actions 등)에서 저장된 상태를 이어받는다. 파일이 없으면 새로 시작."""
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            state = json.load(f)

        for a in self.agents:
            if a.agent_id in state.get("agent_params", {}):
                a.params = state["agent_params"][a.agent_id]

        for aid, pdata in state.get("portfolios", {}).items():
            pf = self.portfolios.get(aid)
            if pf is None:
                continue
            pf.budget = pdata.get("budget", pf.budget)
            pf.cash = pdata["cash"]
            pf.holdings = {k: float(v) for k, v in pdata.get("holdings", {}).items()}
            pf.avg_cost = {k: float(v) for k, v in pdata.get("avg_cost", {}).items()}
            pf.history = pdata.get("history", [])
            pf.peak_nav = pdata.get("peak_nav", pf.budget)
            pf.halt_days_remaining = pdata.get("halt_days_remaining", 0)

        t = state.get("tournament", {})
        self.tournament.points = defaultdict(int, t.get("points", {}))
        self.tournament.week_points = defaultdict(int, t.get("week_points", {}))
        self.tournament.daily_results = [
            DailyResult(date=d["date"], returns=d["returns"], winner=d["winner"])
            for d in t.get("daily_results", [])
        ]
        self.evolution_log = state.get("evolution_log", [])

        ciw = state.get("current_iso_week")
        self._current_iso_week = tuple(ciw) if ciw else None
        self._week_start_nav = state.get("week_start_nav", {})
