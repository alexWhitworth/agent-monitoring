# ruff: noqa: N806
"""Top-level pure orchestration for Tier 1 AMDM detection.

API-009: advance(state, tick) -> (state, TickResult)

Wires Steps 1-6: per-axis updates, S_vector assembly (0.0-fill during
axis warm-up), joint update, TickResult aggregation.
"""

from __future__ import annotations

__all__ = ["advance"]

import numpy as np

from agent_monitor.agg import get_agg
from agent_monitor.axis import update_axis
from agent_monitor.joint import update_joint
from agent_monitor.types import MonitorState, TickMetrics, TickResult


def advance(state: MonitorState, tick: TickMetrics) -> tuple[MonitorState, TickResult]:
    """Run one full tick of Tier 1 AMDM detection.

    Args:
        state: Current MonitorState (tick-level state).
        tick: Per-tick input metric matrix.

    Returns:
        state: Updated MonitorState (all per-axis + joint state advanced).
        result: TickResult for this tick.

    Raises:
        ValueError on shape mismatch or non-finite input.
    """
    cfg = state.config
    A = len(cfg.axis_names)
    X = tick.X

    if len(X) != A:
        raise ValueError(
            f"TickMetrics has {len(X)} axis entries; expected {A}"
        )

    agg_fn = get_agg(cfg.agg)
    axis_results = []
    current_state = state

    for a in range(A):
        current_state, ax_res = update_axis(current_state, a, X[a], agg_fn)  # type: ignore[arg-type]
        axis_results.append(ax_res)

    # Build S_vector: 0.0-fill during axis warm-up (reference parity)
    S_vector = np.array(
        [r.S_A if r.S_A is not None else 0.0 for r in axis_results],
        dtype=np.float64,
    )

    # Joint update
    current_state, joint_result = update_joint(current_state, S_vector)

    any_axis_flag = any(r.flagged for r in axis_results)
    joint_flag = joint_result.flagged
    alert = any_axis_flag or joint_flag

    result = TickResult(
        tick=state.tick,
        axis_results=tuple(axis_results),
        joint=joint_result,
        S_vector=S_vector,
        any_axis_flag=any_axis_flag,
        joint_flag=joint_flag,
        alert=alert,
    )

    # Advance tick counter (mutate via __setattr__ to avoid full copy)
    object.__setattr__(current_state, "tick", current_state.tick + 1)
    return current_state, result
