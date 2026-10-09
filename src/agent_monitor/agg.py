"""Aggregation registry for per-axis z-score composition (Step 2).

API-003: get_agg(name) -> pure function over non-empty z-score sequences.
Empty input raises ValueError (axis warm-up handled upstream).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

__all__ = ["AGGREGATORS", "get_agg"]


def _mean_agg(zs: Sequence[float]) -> float:
    """Arithmetic mean of z-scores. Uses math.fsum for exact summation."""
    if not zs:
        raise ValueError("mean aggregation requires at least one z-score")
    n = len(zs)
    return math.fsum(zs) / n


def _max_agg(zs: Sequence[float]) -> float:
    """Maximum z-score (most sensitive to single-metric spikes)."""
    if not zs:
        raise ValueError("max aggregation requires at least one z-score")
    return max(zs)


def _l2_agg(zs: Sequence[float]) -> float:
    """L2 norm of z-scores: sqrt(sum(z_i^2)). Penalises spread across metrics."""
    if not zs:
        raise ValueError("l2 aggregation requires at least one z-score")
    return math.sqrt(sum(z * z for z in zs))


AGGREGATORS: dict[str, Callable[[Sequence[float]], float]] = {
    "mean": _mean_agg,
    "max": _max_agg,
    "l2": _l2_agg,
}
"""Registry of known aggregation functions."""


def get_agg(name: str) -> Callable[[Sequence[float]], float]:
    """Look up an aggregation function by registry key.

    Raises KeyError on unknown name.
    """
    if name not in AGGREGATORS:
        raise KeyError(
            f"Unknown aggregation '{name}'; must be one of {sorted(AGGREGATORS.keys())}"
        )
    return AGGREGATORS[name]
