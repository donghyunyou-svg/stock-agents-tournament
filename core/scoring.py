"""주간 토너먼트 스코어링: 매일 최고 수익률 에이전트에게 1점, 나머지는 0점."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DailyResult:
    date: str
    returns: Dict[str, float]  # agent_id -> 당일 수익률
    winner: str


class WeeklyTournament:
    def __init__(self):
        self.daily_results: List[DailyResult] = []
        self.points: Dict[str, int] = defaultdict(int)  # 누적 점수
        self.week_points: Dict[str, int] = defaultdict(int)  # 이번 주 점수 (주간 리셋)

    def record_day(self, date: str, daily_returns: Dict[str, float]) -> DailyResult:
        if not daily_returns:
            raise ValueError("daily_returns가 비어 있습니다")
        winner = max(daily_returns.items(), key=lambda x: x[1])[0]
        self.points[winner] += 1
        self.week_points[winner] += 1
        result = DailyResult(date=date, returns=dict(daily_returns), winner=winner)
        self.daily_results.append(result)
        return result

    def weekly_winner(self) -> str:
        if not self.week_points:
            return ""
        return max(self.week_points.items(), key=lambda x: x[1])[0]

    def reset_week(self):
        self.week_points = defaultdict(int)

    def leaderboard(self) -> List[Dict]:
        return sorted(
            [{"agent_id": a, "total_points": p} for a, p in self.points.items()],
            key=lambda x: -x["total_points"],
        )

    def to_dict(self) -> Dict:
        return {
            "points": dict(self.points),
            "week_points": dict(self.week_points),
            "daily_results": [
                {"date": d.date, "returns": d.returns, "winner": d.winner}
                for d in self.daily_results
            ],
        }
