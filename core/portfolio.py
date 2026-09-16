"""가상(모의투자) 포트폴리오/브로커 시뮬레이터."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal

Side = Literal["BUY", "SELL"]


@dataclass
class Order:
    ticker: str
    side: Side
    qty: float  # 주식 수 (분할매수 지원 위해 float)
    reason: str = ""


@dataclass
class Fill:
    ticker: str
    side: Side
    qty: float
    price: float
    fee: float


class VirtualPortfolio:
    """
    한 에이전트가 운용하는 가상 계좌.
    - 초기 예산(budget)으로 시작
    - buy/sell은 즉시 당일 종가로 체결된다고 가정 (모의투자 단순화)
    - fee_rate: 매매 수수료율 (매도 시 세금 포함 근사치)
    """

    def __init__(self, agent_id: str, budget: float, fee_rate: float = 0.00015,
                 tax_rate_sell: float = 0.0018):
        self.agent_id = agent_id
        self.budget = budget
        self.cash = budget
        self.holdings: Dict[str, float] = {}  # ticker -> qty
        self.avg_cost: Dict[str, float] = {}  # ticker -> 평균 매입단가
        self.fee_rate = fee_rate
        self.tax_rate_sell = tax_rate_sell
        self.history: List[Dict] = []  # 일별 NAV 로그
        self.fills: List[Fill] = []
        # --- 리스크 관리(모든 에이전트 공통 안전장치) ---
        self.peak_nav = budget          # 지금까지의 최고 평가금액 (고점 대비 낙폭 계산용)
        self.halt_days_remaining = 0    # 서킷브레이커 발동 시 매수 정지 잔여일

    def execute(self, orders: List[Order], prices: Dict[str, float]) -> List[Fill]:
        fills = []
        for o in orders:
            if o.ticker not in prices or o.qty <= 0:
                continue
            px = prices[o.ticker]
            if o.side == "BUY":
                cost = px * o.qty
                fee = cost * self.fee_rate
                total = cost + fee
                if total > self.cash:
                    # 예산 초과 시 가능한 만큼만 매수 (현금의 99%까지만 사용해 여유 확보)
                    affordable_qty = (self.cash * 0.99) / (px * (1 + self.fee_rate))
                    if affordable_qty <= 0:
                        continue
                    o.qty = affordable_qty
                    cost = px * o.qty
                    fee = cost * self.fee_rate
                    total = cost + fee
                prev_qty = self.holdings.get(o.ticker, 0.0)
                prev_cost = self.avg_cost.get(o.ticker, 0.0)
                new_qty = prev_qty + o.qty
                self.avg_cost[o.ticker] = (
                    (prev_cost * prev_qty + px * o.qty) / new_qty if new_qty > 0 else 0.0
                )
                self.holdings[o.ticker] = new_qty
                self.cash -= total
                fills.append(Fill(o.ticker, "BUY", o.qty, px, fee))
            elif o.side == "SELL":
                held = self.holdings.get(o.ticker, 0.0)
                qty = min(o.qty, held)
                if qty <= 0:
                    continue
                proceeds = px * qty
                fee = proceeds * (self.fee_rate + self.tax_rate_sell)
                self.cash += proceeds - fee
                self.holdings[o.ticker] = held - qty
                if self.holdings[o.ticker] <= 1e-9:
                    del self.holdings[o.ticker]
                    self.avg_cost.pop(o.ticker, None)
                fills.append(Fill(o.ticker, "SELL", qty, px, fee))
        self.fills.extend(fills)
        return fills

    def mark_to_market(self, day: str, prices: Dict[str, float]) -> float:
        equity = sum(prices.get(t, self.avg_cost.get(t, 0.0)) * q
                     for t, q in self.holdings.items())
        nav = self.cash + equity
        self.peak_nav = max(self.peak_nav, nav)
        self.history.append({
            "date": day, "nav": nav, "cash": self.cash, "equity": equity,
            "holdings": dict(self.holdings),
        })
        return nav

    def current_value(self, prices: Dict[str, float]) -> float:
        """주어진(오늘) 가격 기준, 매매 실행 전 현재 평가금액을 계산한다."""
        equity = sum(prices.get(t, self.avg_cost.get(t, 0.0)) * q
                     for t, q in self.holdings.items())
        return self.cash + equity

    def drawdown_from_peak(self, prices: Dict[str, float] = None) -> float:
        """고점(peak_nav) 대비 현재 낙폭 (0.15 = 고점 대비 15% 하락)."""
        if self.peak_nav <= 0:
            return 0.0
        current = self.current_value(prices) if prices is not None else (
            self.history[-1]["nav"] if self.history else self.cash)
        return max(0.0, 1 - current / self.peak_nav)

    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "budget": self.budget,
            "cash": self.cash,
            "holdings": self.holdings,
            "avg_cost": self.avg_cost,
            "history": self.history,
            "peak_nav": self.peak_nav,
            "halt_days_remaining": self.halt_days_remaining,
        }

    def daily_return(self) -> float:
        if len(self.history) < 2:
            return 0.0
        prev = self.history[-2]["nav"]
        cur = self.history[-1]["nav"]
        return (cur / prev - 1) if prev > 0 else 0.0

    def total_return(self) -> float:
        if not self.history:
            return 0.0
        return self.history[-1]["nav"] / self.budget - 1
