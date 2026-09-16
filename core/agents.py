"""
서로 다른 방법론을 사용하는 10개의 투자 에이전트.

각 에이전트는 compute_target_weights()만 구현하면 되고, 공통 로직(BaseAgent.decide)이
목표 비중(target weights)을 실제 매수/매도 주문(Order)으로 변환한다.
params 딕셔너리는 주간 전략 공유(evolution.py)에서 승자 에이전트의 특성을 일부
반영해 조정되는 대상이다 — 단, 각 에이전트의 핵심 정체성(시그널 종류)은 유지된다.
"""

from __future__ import annotations

import abc
from datetime import date
from typing import Dict, List

import pandas as pd

from . import indicators as ind
from .data import Instrument
from .portfolio import Order, VirtualPortfolio


class BaseAgent(abc.ABC):
    agent_id: str
    name: str
    description: str

    def __init__(self, params: Dict = None):
        self.params = params or {}

    @abc.abstractmethod
    def compute_target_weights(
        self, today: date, history: Dict[str, pd.DataFrame],
        macro: Dict[str, float], universe: List[Instrument],
    ) -> Dict[str, float]:
        """티커 -> 목표 비중(0~1, 합계 <=1)을 반환."""

    def decide(
        self, today: date, history: Dict[str, pd.DataFrame],
        macro: Dict[str, float], universe: List[Instrument],
        portfolio: VirtualPortfolio, prices: Dict[str, float],
    ) -> List[Order]:
        target_weights = self.compute_target_weights(today, history, macro, universe)
        return self._rebalance(target_weights, prices, portfolio)

    def _rebalance(self, target_weights: Dict[str, float], prices: Dict[str, float],
                    portfolio: VirtualPortfolio) -> List[Order]:
        equity = sum(prices.get(t, portfolio.avg_cost.get(t, 0.0)) * q
                     for t, q in portfolio.holdings.items())
        nav = portfolio.cash + equity
        orders: List[Order] = []
        # 1) 목표에 없는 종목은 전량 매도
        for ticker in list(portfolio.holdings.keys()):
            if ticker not in target_weights and ticker in prices:
                orders.append(Order(ticker, "SELL", portfolio.holdings[ticker], self.name))
        # 2) 목표 비중과 현재 비중 차이만큼 매수/매도
        for ticker, w in target_weights.items():
            if ticker not in prices:
                continue
            px = prices[ticker]
            target_value = nav * w
            current_qty = portfolio.holdings.get(ticker, 0.0)
            current_value = current_qty * px
            diff_value = target_value - current_value
            if diff_value > px * 0.5:  # 최소 0.5주 이상 차이날 때만 매수
                orders.append(Order(ticker, "BUY", diff_value / px, self.name))
            elif diff_value < -px * 0.5:
                sell_qty = min(current_qty, -diff_value / px)
                if sell_qty > 0:
                    orders.append(Order(ticker, "SELL", sell_qty, self.name))
        return orders


def _top_k(scores: Dict[str, float], k: int, ascending: bool = False) -> List[str]:
    items = sorted(scores.items(), key=lambda x: x[1], reverse=not ascending)
    return [t for t, _ in items[:k]]


class MomentumAgent(BaseAgent):
    agent_id = "agent01_momentum"
    name = "모멘텀 추종 (Momentum)"
    description = "최근 N일 수익률(상대 모멘텀)이 가장 높은 상위 K개 종목에 균등 투자"

    def __init__(self, params=None):
        super().__init__(params or {"window": 60, "top_k": 4})

    def compute_target_weights(self, today, history, macro, universe):
        window, k = self.params["window"], self.params["top_k"]
        scores = {t: ind.momentum_return(df["Close"], window) for t, df in history.items()}
        winners = [t for t in _top_k(scores, k) if scores[t] > 0]
        if not winners:
            return {}
        w = 1.0 / len(winners)
        return {t: w for t in winners}


class MeanReversionAgent(BaseAgent):
    agent_id = "agent02_mean_reversion"
    name = "평균회귀 (Mean Reversion / RSI)"
    description = "RSI가 과매도 구간인 종목을 저가 매수, 과매수 구간이면 회피"

    def __init__(self, params=None):
        super().__init__(params or {"rsi_window": 14, "oversold": 30, "top_k": 4})

    def compute_target_weights(self, today, history, macro, universe):
        rsi_w, thr, k = self.params["rsi_window"], self.params["oversold"], self.params["top_k"]
        scores = {}
        for t, df in history.items():
            r = ind.rsi(df["Close"], rsi_w)
            if not r.empty and pd.notna(r.iloc[-1]) and r.iloc[-1] < thr:
                scores[t] = thr - r.iloc[-1]  # 더 많이 과매도일수록 높은 점수
        if not scores:
            return {}
        winners = _top_k(scores, k)
        w = 1.0 / len(winners)
        return {t: w for t in winners}


class ValueProxyAgent(BaseAgent):
    agent_id = "agent03_value_proxy"
    name = "가치투자 근사 (Value Proxy)"
    description = ("장기(200일) 평균 대비 가장 저평가된(할인폭이 큰) 종목 매수. "
                    "실제 재무데이터(PER/PBR) 연동 전까지는 가격 기반 근사치 사용")

    def __init__(self, params=None):
        super().__init__(params or {"long_window": 200, "top_k": 4})

    def compute_target_weights(self, today, history, macro, universe):
        lw, k = self.params["long_window"], self.params["top_k"]
        scores = {}
        for t, df in history.items():
            sma_long = ind.sma(df["Close"], lw)
            if not sma_long.empty and pd.notna(sma_long.iloc[-1]):
                discount = sma_long.iloc[-1] / df["Close"].iloc[-1] - 1
                if discount > 0:
                    scores[t] = discount
        if not scores:
            return {}
        winners = _top_k(scores, k)
        w = 1.0 / len(winners)
        return {t: w for t in winners}


class MovingAverageCrossAgent(BaseAgent):
    agent_id = "agent04_ma_cross"
    name = "이동평균 골든/데드크로스 (Trend Following)"
    description = "단기 이동평균이 장기 이동평균을 상향 돌파(골든크로스)한 종목 매수"

    def __init__(self, params=None):
        super().__init__(params or {"short": 20, "long": 60, "top_k": 4})

    def compute_target_weights(self, today, history, macro, universe):
        s, l, k = self.params["short"], self.params["long"], self.params["top_k"]
        scores = {}
        for t, df in history.items():
            sma_s, sma_l = ind.sma(df["Close"], s), ind.sma(df["Close"], l)
            if pd.notna(sma_s.iloc[-1]) and pd.notna(sma_l.iloc[-1]) and sma_s.iloc[-1] > sma_l.iloc[-1]:
                scores[t] = sma_s.iloc[-1] / sma_l.iloc[-1] - 1
        if not scores:
            return {}
        winners = _top_k(scores, k)
        w = 1.0 / len(winners)
        return {t: w for t in winners}


class MACDAgent(BaseAgent):
    agent_id = "agent05_macd"
    name = "MACD 시그널"
    description = "MACD선이 시그널선을 상향 돌파한 종목 매수"

    def __init__(self, params=None):
        super().__init__(params or {"fast": 12, "slow": 26, "signal": 9, "top_k": 4})

    def compute_target_weights(self, today, history, macro, universe):
        p = self.params
        scores = {}
        for t, df in history.items():
            macd_line, signal_line, hist = ind.macd(df["Close"], p["fast"], p["slow"], p["signal"])
            if len(hist) >= 2 and pd.notna(hist.iloc[-1]) and pd.notna(hist.iloc[-2]):
                if hist.iloc[-1] > 0 and hist.iloc[-2] <= 0:  # 막 상향 돌파
                    scores[t] = hist.iloc[-1]
                elif hist.iloc[-1] > 0:
                    scores[t] = hist.iloc[-1] * 0.3  # 이미 상승 중이면 낮은 가중
        if not scores:
            return {}
        winners = _top_k(scores, p["top_k"])
        w = 1.0 / len(winners)
        return {t: w for t in winners}


class VolumeBreakoutAgent(BaseAgent):
    agent_id = "agent06_volume_breakout"
    name = "거래량 돌파 (Volume Breakout)"
    description = "N일 신고가를 평균 대비 급증한 거래량과 함께 돌파한 종목 매수"

    def __init__(self, params=None):
        super().__init__(params or {"lookback": 20, "vol_mult": 1.5, "top_k": 4})

    def compute_target_weights(self, today, history, macro, universe):
        lb, mult, k = self.params["lookback"], self.params["vol_mult"], self.params["top_k"]
        scores = {}
        for t, df in history.items():
            if len(df) < lb + 1:
                continue
            recent_high = df["Close"].iloc[-lb - 1:-1].max()
            avg_vol = df["Volume"].iloc[-lb - 1:-1].mean()
            last_close, last_vol = df["Close"].iloc[-1], df["Volume"].iloc[-1]
            if last_close > recent_high and last_vol > avg_vol * mult:
                scores[t] = (last_close / recent_high - 1) * (last_vol / max(avg_vol, 1))
        if not scores:
            return {}
        winners = _top_k(scores, k)
        w = 1.0 / len(winners)
        return {t: w for t in winners}


class MacroSectorRotationAgent(BaseAgent):
    agent_id = "agent07_macro_rotation"
    name = "거시경제 기반 섹터 로테이션"
    description = "금리/환율/유가 등 거시 지표로 우호적인 섹터를 선별해 비중 조절"

    def __init__(self, params=None):
        super().__init__(params or {"top_k": 4, "risk_off_yield": 4.5})

    def compute_target_weights(self, today, history, macro, universe):
        k = self.params["top_k"]
        yield_10y = macro.get("us_10y_yield", 4.0)
        oil = macro.get("wti_oil", 75)
        risk_off = yield_10y > self.params["risk_off_yield"]
        # 고유가 -> 에너지 섹터 선호, 금리 상승(risk-off) -> 전체 비중 축소
        preferred_sectors = {"Energy"} if oil > 80 else set()
        scores = {}
        for inst in universe:
            if inst.ticker not in history:
                continue
            mom = ind.momentum_return(history[inst.ticker]["Close"], 20)
            bonus = 0.05 if inst.sector in preferred_sectors else 0.0
            scores[inst.ticker] = mom + bonus
        winners = _top_k(scores, k)
        winners = [t for t in winners if scores[t] > -0.5]
        if not winners:
            return {}
        exposure = 0.5 if risk_off else 1.0  # risk-off 시 현금 비중 확대
        w = exposure / len(winners)
        return {t: w for t in winners}


class VolatilityTargetingAgent(BaseAgent):
    agent_id = "agent08_vol_targeting"
    name = "변동성 타겟팅 (Inverse Volatility)"
    description = "종목별 변동성에 반비례하여 비중을 배분해 포트폴리오 리스크를 평준화"

    def __init__(self, params=None):
        super().__init__(params or {"vol_window": 20, "top_k": 6})

    def compute_target_weights(self, today, history, macro, universe):
        vw, k = self.params["vol_window"], self.params["top_k"]
        vols = {t: ind.volatility(df["Close"], vw) for t, df in history.items()}
        vols = {t: v for t, v in vols.items() if v > 1e-6}
        if not vols:
            return {}
        chosen = _top_k(vols, k, ascending=True)  # 변동성 낮은 순
        inv_vol = {t: 1.0 / vols[t] for t in chosen}
        total = sum(inv_vol.values())
        return {t: v / total for t, v in inv_vol.items()}


class DualMomentumAgent(BaseAgent):
    agent_id = "agent09_dual_momentum"
    name = "듀얼 모멘텀 (절대+상대)"
    description = "절대 모멘텀(자체 수익률 양전환)과 상대 모멘텀(동종 대비 우위)을 동시 만족하는 종목 매수"

    def __init__(self, params=None):
        super().__init__(params or {"window": 90, "top_k": 3})

    def compute_target_weights(self, today, history, macro, universe):
        window, k = self.params["window"], self.params["top_k"]
        scores = {t: ind.momentum_return(df["Close"], window) for t, df in history.items()}
        positive = {t: s for t, s in scores.items() if s > 0}  # 절대 모멘텀 필터
        if not positive:
            return {}
        winners = _top_k(positive, k)  # 상대 모멘텀 필터(상위 K)
        w = 1.0 / len(winners)
        return {t: w for t in winners}


class BuyHoldBenchmarkAgent(BaseAgent):
    agent_id = "agent10_buy_hold"
    name = "매수 후 보유 (벤치마크)"
    description = "전 종목 균등비중으로 최초 1회 매수 후 그대로 보유하는 벤치마크 에이전트"

    def __init__(self, params=None):
        super().__init__(params or {})
        self._bought = False

    def compute_target_weights(self, today, history, macro, universe):
        tickers = [t for t in history.keys()]
        if not tickers:
            return {}
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}


AGENT_CLASSES = [
    MomentumAgent, MeanReversionAgent, ValueProxyAgent, MovingAverageCrossAgent,
    MACDAgent, VolumeBreakoutAgent, MacroSectorRotationAgent, VolatilityTargetingAgent,
    DualMomentumAgent, BuyHoldBenchmarkAgent,
]


def build_default_agents() -> List[BaseAgent]:
    return [cls() for cls in AGENT_CLASSES]
