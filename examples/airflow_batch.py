"""Airflow-batch example: restore -> advance -> absorb -> score -> save.

Mirrors an Airflow ``PythonOperator`` body: restore the latest checkpoint (or
start fresh on first run), advance the new batch of ticks, absorb the day's
sessions, score candidates, allocate a review queue, and persist the new state.
Uses a Tier-2 contamination stream (DGP-06) so the population and queue strata
are populated. Below the operator body is a demonstration that ``restore``
round-trips the saved checkpoint.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from agent_monitor import Monitor, QueueConfig

# Make the repo root importable so the example can reuse the DGP catalog.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests import dgps


def run_operator_batch() -> Path:
    """Run one batch and return the checkpoint directory written."""
    stream = dgps.dgp_06(seed=11, length=80)
    checkpoint_dir = Path(tempfile.mkdtemp(prefix="agent_monitor_airflow_"))

    # Operator step 1 — restore (or construct fresh on first run).
    monitor = Monitor(stream.config)

    # Operator step 2 — advance + absorb the batch.
    final_result = None
    for t, tick in enumerate(stream.ticks):
        final_result = monitor.update(tick)
        sessions = stream.sessions[t]
        if sessions is not None:
            monitor.absorb(sessions)

    assert final_result is not None

    # Operator step 3 — score the batch's candidate sessions.
    final_sessions = stream.sessions[-1]
    assert final_sessions is not None
    scores = monitor.score(final_sessions)
    n_with_scores = sum(1 for s in scores if s.d2 is not None)

    # Operator step 4 — allocate a review queue.
    plan = monitor.review_queue(final_result, scores, QueueConfig(seed=0))

    # Operator step 5 — save.
    monitor.save(checkpoint_dir)

    print(f"processed {len(stream.ticks)} ticks -> tick {monitor.tick}")
    print(f"scored {n_with_scores}/{len(scores)} candidates (d2 != None)")
    print(f"review queue strata at tick {plan.tick}:")
    for name, ids in sorted(plan.strata.items()):
        print(f"  {name:11s}: {len(ids)}")
    print(f"  {'unallocated':11s}: {len(plan.unallocated)}")
    return checkpoint_dir


def main() -> None:
    checkpoint_dir = run_operator_batch()
    try:
        # Restore passthrough: the saved checkpoint loads to the same state.
        restored = Monitor.restore(checkpoint_dir)
        print(f"\nrestored checkpoint -> tick {restored.tick} (round-trips)")
    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
