# ruff: noqa: N806
"""Review queue stratification (F-009, API-014).

Five-stratum allocation from current-tick candidates in priority order
(top_k > flagged > per_axis > uniform > reserved). Disjoint membership;
min(budget, available) with shortfall cascade within the same build call;
seeded RNG; reserved never filled.
"""

from __future__ import annotations

__all__ = ["build_review_queue"]

import numpy as np

from agent_monitor.types import (
    MonitorState,
    QueueConfig,
    ReviewQueuePlan,
    SessionScore,
    TickResult,
)


def build_review_queue(
    tick_result: TickResult,
    session_scores: tuple[SessionScore, ...],
    state: MonitorState,
    qcfg: QueueConfig,
) -> ReviewQueuePlan:
    """Five-stratum allocation from current-tick candidates.

    Priority order: top_k > flagged > per_axis > uniform > reserved.
    Shortfall from any stratum cascades to the next within the same build call.
    'reserved' is always empty (library never fills it).

    Args:
        tick_result: Tier 1 result for the current tick.
        session_scores: Scored candidate sessions (d2 may be None during warm-up).
        state: Current monitor state (for population feature/flagged lookups).
        qcfg: Queue budgets and RNG seed.

    Returns:
        ReviewQueuePlan with disjoint strata and unallocated remainder.
    """
    if not session_scores:
        return ReviewQueuePlan(
            tick=tick_result.tick,
            strata={"top_k": (), "flagged": (), "per_axis": (), "uniform": (), "reserved": ()},
            unallocated=(),
        )

    S = len(session_scores)
    A = len(state.config.axis_names)

    # Extract candidate metadata
    cand_ids = [s.id for s in session_scores]
    # Treat None d2 as -inf so they sort to the bottom (won't be selected in top_k)
    cand_d2 = np.array(
        [s.d2 if s.d2 is not None else -np.inf for s in session_scores], dtype=np.float64
    )

    # Build population lookups
    pop = state.population
    pop_id_to_idx: dict[str, int] = {}
    pop_id_to_flagged: dict[str, bool] = {}
    for i in range(len(pop.ids)):
        sid = str(pop.ids[i])  # Ensure string key
        pop_id_to_idx[sid] = i
        pop_id_to_flagged[sid] = bool(pop.tick_flagged[i])

    # Look up candidate features and flagged status from population
    cand_features = np.zeros((S, A), dtype=np.float64)
    cand_flagged = np.zeros(S, dtype=bool)
    for i, sc in enumerate(session_scores):
        idx = pop_id_to_idx.get(sc.id)
        if idx is not None:
            cand_features[i] = pop.features[idx]
        cand_flagged[i] = pop_id_to_flagged.get(sc.id, False)

    # Track allocations
    allocated: set[int] = set()
    strata: dict[str, tuple[str, ...]] = {}

    # D2-based sort helper (descending, unallocated only)
    def _top_by_d2(pool: list[int], k: int) -> list[int]:
        sorted_pool = sorted(pool, key=lambda i: cand_d2[i], reverse=True)
        return sorted_pool[:k]

    # ---- Stratum 1: top_k ----
    remaining = [i for i in range(S) if i not in allocated]
    top_k_selected = _top_by_d2(remaining, qcfg.top_k)
    allocated.update(top_k_selected)
    strata["top_k"] = tuple(cand_ids[i] for i in top_k_selected)
    cascade = max(0, qcfg.top_k - len(top_k_selected))

    # ---- Stratum 2: flagged ----
    remaining = [i for i in range(S) if i not in allocated]
    flagged_pool = [i for i in remaining if cand_flagged[i]]
    flagged_budget = qcfg.flagged + cascade
    flagged_selected = _top_by_d2(flagged_pool, flagged_budget)
    allocated.update(flagged_selected)
    strata["flagged"] = tuple(cand_ids[i] for i in flagged_selected)
    cascade = max(0, cascade + qcfg.flagged - len(flagged_selected))

    # ---- Stratum 3: per_axis ----
    remaining = [i for i in range(S) if i not in allocated]
    per_axis_budget = qcfg.per_axis + cascade
    per_axis_each = per_axis_budget // A if A > 0 else 0

    per_axis_selected: set[int] = set()
    for ax in range(A):
        if per_axis_each <= 0:
            break
        pool = [i for i in remaining if i not in per_axis_selected]
        # Sort by absolute feature value on this axis, descending
        sorted_ax = sorted(pool, key=lambda i: abs(cand_features[i, ax]), reverse=True)
        take = min(per_axis_each, len(sorted_ax))
        per_axis_selected.update(sorted_ax[:take])

    allocated.update(per_axis_selected)
    strata["per_axis"] = tuple(cand_ids[i] for i in per_axis_selected)
    cascade = max(0, cascade + qcfg.per_axis - len(per_axis_selected))

    # ---- Stratum 4: uniform ----
    remaining = [i for i in range(S) if i not in allocated]
    uniform_budget = min(qcfg.uniform + cascade, len(remaining))
    rng = np.random.default_rng(qcfg.seed)
    uniform_selected = list(
        rng.choice(remaining, size=uniform_budget, replace=False)
    ) if uniform_budget > 0 else []
    allocated.update(uniform_selected)
    strata["uniform"] = tuple(cand_ids[i] for i in uniform_selected)

    # ---- Stratum 5: reserved ----
    strata["reserved"] = ()

    # Unallocated remainder
    unallocated = tuple(cand_ids[i] for i in range(S) if i not in allocated)

    return ReviewQueuePlan(
        tick=tick_result.tick,
        strata=strata,
        unallocated=unallocated,
    )