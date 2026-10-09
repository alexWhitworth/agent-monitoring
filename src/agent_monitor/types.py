"""Immutable data schemas for the AMDM pipeline.

All schemas use @dataclass(frozen=True).  NumPy arrays are marked read-only
in __post_init__ to enforce the immutability contract.
"""

from __future__ import annotations

__all__ = [
    "AxisResult",
    "JointResult",
    "MonitorConfig",
    "MonitorState",
    "QueueConfig",
    "ReviewQueuePlan",
    "SessionFeatures",
    "SessionScore",
    "SessionWindow",
    "TickMetrics",
    "TickResult",
]

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Construction-time configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitorConfig:
    """Frozen construction-time configuration; stored in every state snapshot.

    Paper defaults: w=80, lambda=0.25, k=None (resolves to chi2_A(0.99)),
    alpha=0.01, min_warm=30, ridge=1e-6, agg='mean'.
    """

    axis_names: tuple[str, ...]
    """Tuple of A >= 2 axis names."""

    metric_names: tuple[tuple[str, ...], ...]
    """>= 1 metric name per axis; len == A."""

    window: int = 80
    """Rolling window length w >= 2."""

    lam: float = 0.25
    """EWMA smoothing factor, 0 < lam <= 1."""

    k: float | None = None
    """Per-axis sensitivity multiplier; None resolves to chi2.ppf(0.99, df=A)."""

    alpha: float = 0.01
    """Joint false-alarm rate."""

    min_warm: int = 30
    """Joint detector warm-up ticks, >= 2."""

    ridge: float = 1e-6
    """Covariance regularization epsilon."""

    agg: str = "mean"
    """Aggregation registry key: 'mean' | 'max' | 'l2'."""

    population_window: int = 80
    """Tier 2 covariance window in ticks, >= 1."""

    min_population: int = 30
    """Sessions required before Tier 2 scoring."""

    winsor_pct: float | None = 0.99
    """Per-axis clip percentile, two-sided [q_{1-p}, q_p]; None disables."""

    winsor_k: int = 3
    """Quantile lookback multiplier: retention = max(population_window, window * winsor_k)."""


@dataclass(frozen=True)
class QueueConfig:
    """Frozen review-queue budgets.

    Defaults (200/150/100/100/50) assume one build per day; per-tick users must scale.
    """

    top_k: int = 200
    """Stratum 1: highest d² sessions."""

    flagged: int = 150
    """Stratum 2: sessions from AMDM-flagged ticks."""

    per_axis: int = 100
    """Stratum 3: floor(per_axis / A) per axis, most extreme first."""

    uniform: int = 100
    """Stratum 4: seeded uniform random from remainder."""

    reserved: int = 50
    """Stratum 5: analyst reserve; library never fills it."""

    seed: int = 0
    """RNG seed for reproducible random strata."""


# ---------------------------------------------------------------------------
# Immutable runtime state
# ---------------------------------------------------------------------------


def _freeze_arrays(*arrs: np.ndarray | None) -> None:
    """Mark the given ndarrays read-only (best-effort on None entries)."""
    for a in arrs:
        if a is not None:
            a.flags.writeable = False


@dataclass(frozen=True)
class SessionWindow:
    """Immutable Tier 2 population store.

    Retains rows for max(population_window, window * winsor_k) ticks.
    Covariance uses the last population_window ticks; winsor quantiles
    use the full retention.
    """

    ids: np.ndarray
    """(R,) read-only unique session ids."""

    ticks: np.ndarray
    """(R,) int64 read-only tick of each row."""

    features: np.ndarray
    """(R, A) float64 read-only raw feature values."""

    tick_flagged: np.ndarray
    """(R,) bool read-only: was the row's tick AMDM-flagged."""

    absorbed_at: np.ndarray
    """(R,) int64 read-only tick at which the row entered the window."""

    def __post_init__(self) -> None:
        _freeze_arrays(
            self.ids, self.ticks, self.features, self.tick_flagged, self.absorbed_at
        )


@dataclass(frozen=True)
class MonitorState:
    """Frozen checkpointable state value.

    All NumPy arrays are float64/int64, marked read-only.
    """

    config: MonitorConfig
    """Fixed topology."""

    tick: int
    """Global tick counter, 0-based."""

    metric_buf: np.ndarray
    """(A, M_max, w) float64 read-only per-metric ring buffers."""

    metric_count: np.ndarray
    """(A, M_max) int64 read-only per-metric valid sample counts."""

    theta: np.ndarray | None
    """(A,) float64 EWMA baselines | None until first S_A."""

    axis_buf: np.ndarray
    """(A, w) float64 read-only S_A history ring."""

    axis_count: np.ndarray
    """(A,) int64 read-only S_A history counts."""

    n: int
    """Welford sample count for joint detector."""

    mu: np.ndarray
    """(A,) float64 Welford joint mean."""

    M2: np.ndarray
    """(A, A) float64 Welford scatter matrix."""

    population: SessionWindow
    """Session rows retained for Tier 2."""

    def __post_init__(self) -> None:
        _freeze_arrays(
            self.metric_buf,
            self.metric_count,
            self.theta,
            self.axis_buf,
            self.axis_count,
            self.mu,
            self.M2,
        )
        # population has its own __post_init__ freeze.
        # SessionWindow is already frozen.


# ---------------------------------------------------------------------------
# Per-tick inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TickMetrics:
    """Per-tick input: one complete, caller-aligned metric matrix (Tier 1)."""

    X: tuple[np.ndarray, ...]
    """Per-axis float arrays, A entries total (ragged-safe M_a per axis); finite float64."""

    def __post_init__(self) -> None:
        _freeze_arrays(*self.X)


@dataclass(frozen=True)
class SessionFeatures:
    """Tier 2 candidate sessions for one tick; features are raw caller-supplied values."""

    ids: tuple[str, ...]
    """(S,) string session ids, unique within the batch."""

    tick: int
    """Tick the sessions belong to."""

    F: np.ndarray
    """(S, A) float64 raw per-axis feature values."""

    def __post_init__(self) -> None:
        _freeze_arrays(self.F)


# ---------------------------------------------------------------------------
# Per-tick outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AxisResult:
    """Per-axis detection output for one tick (Steps 1-4)."""

    axis: str
    """Axis name."""

    S_A: float | None
    """Axis score; None during z-score warm-up."""

    theta: float | None
    """EWMA baseline."""

    sigma_S: float | None  # noqa: N815
    """Rolling std of S_A over full window."""

    deviation: float | None
    """|S_A - theta_A|."""

    threshold: float | None
    """k * sigma_S."""

    flagged: bool
    """False whenever sigma_S <= 1e-10 or warm-up."""

    note: str
    """Warm-up/status explanation."""


@dataclass(frozen=True)
class JointResult:
    """Joint Mahalanobis/chi-square detection output (Steps 5-6)."""

    n: int
    """Welford count."""

    D2: float | None
    """Squared Mahalanobis distance; None during warm-up."""

    threshold: float
    """chi2.ppf(1-alpha, A)."""

    p_value: float | None
    """chi2.sf(D2, A)."""

    flagged: bool
    """D2 > threshold."""

    note: str
    """Warm-up/status explanation."""


@dataclass(frozen=True)
class TickResult:
    """Complete Tier 1 output for one tick."""

    tick: int
    """Tick index."""

    axis_results: tuple[AxisResult, ...]
    """Length A."""

    joint: JointResult
    """Joint detection result."""

    S_vector: np.ndarray | None
    """(A,) axis scores; None entries filled 0.0 during axis warm-up."""

    any_axis_flag: bool
    """Any per-axis flag."""

    joint_flag: bool
    """Joint flag."""

    alert: bool
    """any_axis_flag or joint_flag."""


# ---------------------------------------------------------------------------
# Tier 2 outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionScore:
    """Tier 2 score for one candidate session."""

    id: str
    """Session id."""

    d2: float | None
    """Squared Mahalanobis distance vs winsorized population; None during warm-up."""

    p_value: float | None
    """chi2.sf(d2, A); nominal (approximate under winsorization)."""


@dataclass(frozen=True)
class ReviewQueuePlan:
    """Stratified human-review allocation for one build call."""

    tick: int
    """Tick the plan was built for."""

    strata: Mapping[str, tuple[str, ...]]
    """Stratum name -> disjoint session ids; 'reserved' key present and always empty."""

    unallocated: tuple[str, ...]
    """Candidate ids not selected."""
