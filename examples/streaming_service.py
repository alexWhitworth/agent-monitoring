"""Streaming-service example: a long-lived, in-memory Monitor over a synthetic
subsequence-burst anomaly stream.

Drives the public ``agent_monitor.Monitor`` API tick by tick: ``update()`` each
tick, ``absorb()`` each tick's sessions into the Tier-2 population, then after
warm-up scores the final tick's sessions and allocates a stratified review
queue. Synthetic data comes from the DGP catalog (``tests/dgps.py``), which is
importable here without becoming part of ``agent_monitor``'s public API.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent_monitor import Monitor, QueueConfig

# Make the repo root importable so the example can reuse the DGP catalog.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests import dgps


def main() -> None:
    # DGP-05: single-axis +3 sigma burst for 3 ticks (BASELINE_TICKS onward).
    stream = dgps.dgp_05(seed=7, length=300)
    monitor = Monitor(stream.config)

    print(f"stream: {stream.taxonomy} ({stream.dgp_id})")
    print(f"declared: {stream.declared}\n")

    alert_ticks: list[int] = []
    final_result = None
    for t, tick in enumerate(stream.ticks):
        result = monitor.update(tick)
        sessions = stream.sessions[t]
        if sessions is not None:
            monitor.absorb(sessions)
        if result.alert:
            alert_ticks.append(t)
            print(
                f"tick {t:3d}: ALERT  axis_flag={result.any_axis_flag} "
                f"joint_flag={result.joint_flag}"
            )
        final_result = result

    assert final_result is not None
    print(f"\n{alert_ticks} alerting tick(s)")

    # After warm-up, score the final tick's sessions and allocate a queue.
    final_sessions = stream.sessions[-1]
    assert final_sessions is not None
    scores = monitor.score(final_sessions)
    plan = monitor.review_queue(final_result, scores, QueueConfig(seed=0))

    print(f"\nreview queue at tick {plan.tick}:")
    for name, ids in sorted(plan.strata.items()):
        print(f"  {name:11s}: {len(ids)}")
    print(f"  {'unallocated':11s}: {len(plan.unallocated)}")


if __name__ == "__main__":
    main()
