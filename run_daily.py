"""
GitHub Actions에서 매일 실행되는 진입점 스크립트.

1) state/live_state.json에 저장된 이전 상태를 불러오고
2) KIS Open API로 오늘 시점 실데이터를 가져와 10개 에이전트를 한 사이클 실행하고
3) 결과를 다시 state/live_state.json에 저장한다 (그 다음 워크플로우가 git commit/push).

환경변수(GitHub Actions Secrets에서 주입):
  KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT_NO
"""

import os
import sys
from datetime import date, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data import KISOpenAPIProvider
from core.simulator import TournamentSimulator

STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state", "live_state.json")
KST = timezone(timedelta(hours=9))


def main():
    app_key = os.environ.get("KIS_APP_KEY", "")
    app_secret = os.environ.get("KIS_APP_SECRET", "")
    account_no = os.environ.get("KIS_ACCOUNT_NO", "")
    if not (app_key and app_secret and account_no):
        print("KIS_APP_KEY / KIS_APP_SECRET / KIS_ACCOUNT_NO 환경변수가 필요합니다.", file=sys.stderr)
        sys.exit(1)

    # 모의투자 여부: 실전 전환 전까지는 항상 True로 유지
    paper = os.environ.get("KIS_PAPER", "true").lower() != "false"

    provider = KISOpenAPIProvider(app_key, app_secret, account_no, paper=paper)
    sim = TournamentSimulator(provider, budget_per_agent=10_000_000)
    sim.load_state(STATE_PATH)

    from datetime import datetime
    today = datetime.now(KST).date()

    print(f"=== {today.isoformat()} 실행 (paper={paper}) ===")
    result = sim.run_day(today)
    if result is None:
        print("오늘은 유효한 시세 데이터가 없습니다 (휴장일이거나 API 오류). 상태를 변경하지 않고 종료합니다.")
        return

    print(f"오늘의 승자: {result.winner} (수익률 {result.returns[result.winner] * 100:+.2f}%)")
    summary = sim.summary()
    print("\n--- 누적 리더보드 ---")
    for row in summary["leaderboard"]:
        print(f"  {row['agent_id']:>26s} : {row['total_points']}점")

    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    sim.save_state(STATE_PATH)
    print(f"\n상태 저장 완료: {STATE_PATH}")


if __name__ == "__main__":
    main()
