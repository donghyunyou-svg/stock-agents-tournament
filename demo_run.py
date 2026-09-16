"""
데모/검증 스크립트: SyntheticDataProvider(가상 시세)로 약 3주(15 거래일)를 시뮬레이션해
전체 파이프라인(데이터->10개 에이전트 판단->가상 체결->일별 스코어링->주간 전략공유(진화))이
정상 동작하는지 검증한다.

실제 운영 시에는 SyntheticDataProvider 대신 FDRProvider/YFinanceProvider/
MeritzOpenAPIProvider로 교체한다 (core/data.py 참고).
"""
import json
from datetime import date

import pandas as pd

from core.data import SyntheticDataProvider
from core.simulator import TournamentSimulator

provider = SyntheticDataProvider(seed=7)
sim = TournamentSimulator(provider, budget_per_agent=10_000_000)

end = date(2026, 9, 15)
trading_days = [d.date() for d in pd.bdate_range(end=end, periods=15)]

for d in trading_days:
    result = sim.run_day(d)
    if result:
        print(f"{result.date} | winner: {result.winner:>22s} | "
              f"return: {result.returns[result.winner]*100:+.2f}%")

summary = sim.summary()
print("\n=== 누적 리더보드 (일별 1위 획득 점수) ===")
for row in summary["leaderboard"]:
    print(f"  {row['agent_id']:>26s} : {row['total_points']}점")

print("\n=== 최종 NAV / 총수익률 ===")
for aid, info in sorted(summary["nav_summary"].items(), key=lambda x: -x[1]["total_return_pct"]):
    print(f"  {info['name']:>28s} : {info['total_return_pct']:+.2f}%  (NAV={info['final_nav']:,.0f})")

print("\n=== 주간 전략 공유(진화) 로그 ===")
for log in summary["evolution_log"]:
    print(f"- {log['week']} 주간 우승: {log['winner']} (주간수익률 {log.get('winner_weekly_return', 0)*100:.2f}%)")
    for ch in log["changes"]:
        print(f"    {ch['agent_id']}.{ch['param']}: {ch['old']} -> {ch['new']} ({ch['reason']})")

sim.save_state("state/demo_state.json")
with open("state/demo_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
print("\n상태 저장 완료: state/demo_state.json, state/demo_summary.json")
