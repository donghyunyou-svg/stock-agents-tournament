"""
주간 전략 공유(진화) 로직.

매주 가장 좋은 성과를 낸 에이전트의 '방법'(어떤 파라미터로 어떤 신호가 잘 통했는지)을
요약해 모든 에이전트에게 공유한다. 각 에이전트는:
  1) 자신의 핵심 정체성(시그널 종류: 모멘텀/평균회귀/가치 등)은 유지하면서,
  2) 승자와 공유되는 '숫자형 파라미터'(window, top_k 등)만 학습률(learning_rate)만큼
     승자 쪽으로 소폭 이동시켜 다음 주 경쟁에 반영한다.
  3) 승자 자신은 이번 주 결과를 기준으로 약간의 탐험(exploration)을 위해 top_k를
     미세하게 흔들어 지역 최적점에 고착되는 것을 방지한다.

이렇게 하면 "승자 전략을 그대로 복제"하지 않고, 각 에이전트가 자신만의 방법론 안에서
승자의 통찰을 참고해 스스로 발전시키는 방식이 된다.
"""

from __future__ import annotations

import random
from typing import Dict, List

from .agents import BaseAgent

# 진화 대상이 되는 숫자형 파라미터 키 (여러 에이전트가 공유하는 개념들)
SHARED_NUMERIC_KEYS = ["window", "top_k", "long_window", "short", "long",
                       "vol_window", "lookback", "rsi_window"]

LEARNING_RATE = 0.15  # 승자 방향으로 이동하는 비율
EXPLORATION_STD = 0.5  # 승자의 탐험적 파라미터 흔들림 정도 (top_k 기준)


def share_and_evolve(agents: List[BaseAgent], weekly_returns: Dict[str, float],
                      week_label: str, rng: random.Random = None) -> Dict:
    """
    weekly_returns: agent_id -> 이번 주 누적 수익률
    반환: 이번 주 공유된 내용을 기록한 로그(dict) — Project 문서화에 사용
    """
    rng = rng or random.Random()
    if not weekly_returns:
        return {"week": week_label, "winner": None, "changes": []}

    winner_id = max(weekly_returns.items(), key=lambda x: x[1])[0]
    winner = next((a for a in agents if a.agent_id == winner_id), None)
    if winner is None:
        return {"week": week_label, "winner": None, "changes": []}

    changes = []
    for agent in agents:
        if agent.agent_id == winner_id:
            # 승자: 탐험적으로 top_k를 소폭 흔들어 다음 주에도 계속 개선 여지를 둠
            if "top_k" in agent.params:
                old = agent.params["top_k"]
                delta = rng.choice([-1, 0, 1])
                new = max(1, min(8, old + delta))
                if new != old:
                    agent.params["top_k"] = new
                    changes.append({
                        "agent_id": agent.agent_id, "param": "top_k",
                        "old": old, "new": new, "reason": "승자 탐험적 조정",
                    })
            continue

        for key in SHARED_NUMERIC_KEYS:
            if key in agent.params and key in winner.params:
                old = agent.params[key]
                target = winner.params[key]
                if old == target:
                    continue
                new = old + (target - old) * LEARNING_RATE
                # top_k류는 정수 유지
                if isinstance(old, int):
                    new = int(round(new))
                if new != old:
                    agent.params[key] = new
                    changes.append({
                        "agent_id": agent.agent_id, "param": key,
                        "old": old, "new": new,
                        "reason": f"주간 우승 에이전트({winner_id})의 {key} 방향으로 {LEARNING_RATE:.0%} 이동",
                    })

    return {
        "week": week_label,
        "winner": winner_id,
        "winner_weekly_return": weekly_returns[winner_id],
        "winner_params_shared": dict(winner.params),
        "changes": changes,
    }
