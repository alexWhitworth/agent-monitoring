"""Seeded anomaly-type DGP catalog (F-014).

Six deterministic, pure data-generating processes that emit full-lifecycle
``TickMetrics`` / ``SessionFeatures`` streams and declare their expected-flag
semantics, mapped to the Schmidl, Wenig & Papenbrock (2022) time-series anomaly
taxonomy (VLDB 15(9), DOI 10.14778/3538598.3538602):

    DGP-01  H0 iid baseline                    "point (none)"
    DGP-02  point anomaly                      single-axis x20-sigma, 1 tick
    DGP-03  multivariate-coordinated           all axes +1.5-sigma, 3 ticks (joint only)
    DGP-04  ramp (characterization only)       single-axis +0.05-sigma/tick drift
    DGP-05  subsequence burst                  single-axis +3-sigma, 3 ticks
    DGP-06  Tier-2 population contamination    0.5% outlier sessions in the winsor tail

Determinism invariant [G1]: tick ``t`` is a pure function of ``(dgp_id, seed,
t)``; the full stream is drawn from one ``numpy.random.default_rng(seed)`` in a
fixed draw order, so the same ``(dgp_id, seed, length)`` yields byte-identical
streams and differing seeds yield differing streams.

Calibration note (required for the [G2] semantics)
--------------------------------------------------
The per-axis layer flags iff ``|S_A - theta| > k * sigma_S``.  A single-tick
spike of raw magnitude ``s`` surfaces as an axis score
``S_A = s / sqrt(1 + s^2/w)`` (the rolling z-score normalizes by a window std
that already contains the spike), so ``S_A`` is capped at ``sqrt(w)``.  The
rolling std of ``S_A`` is ``sigma_S >= |S_A| / sqrt(w)``, so the per-axis layer
can only ever fire when ``k < sqrt(w)``.  The paper default
``k = chi2_A(0.99) >= chi2_2(0.99) ≈ 9.21`` exceeds ``sqrt(80) ≈ 8.94`` at the
default window, making the per-axis layer *provably inert* for any A >= 2
(the reference implementation's own demo observes this: its "x20 sigma" spike
is absorbed before ``|dev|`` can exceed ``k * sigma_S``).

To exercise the per-axis layer honestly, the catalog therefore sets
``k = 7.0 < sqrt(250) ≈ 15.8`` and widens the metric count to ``M = 25`` and the
window to ``w = 250``.  These values were chosen so that, in the catalog's fixed
seeds, the +3-sigma subsequence (DGP-05) crosses ``k * sigma_S`` at onset while
the +1.5-sigma coordinated shift (DGP-03) does not (each marginal stays within
``k``), leaving only the joint layer to fire for DGP-03.  This is a
*catalog-level* sensitivity choice, not a change to the library: ``k``, ``w``
and the metric count are all construction-time ``MonitorConfig`` fields.

Out of scope for v0
-------------------
A seasonal/contextual DGP is deliberately excluded from v0 pass/fail: periodicity
under EWMA baselines + rolling z-scores is a calibration question deferred to v1
with real data (risk R-04).  DGP-04 documents, but does not assert on, flag
behavior for the same reason.

Streams shorter than the anomaly onset (``length <= BASELINE_TICKS``) emit pure
warm-up/baseline data only; no anomaly is injected and no error is raised.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from agent_monitor.config import validate_monitor_config
from agent_monitor.types import MonitorConfig, SessionFeatures, TickMetrics

# ---------------------------------------------------------------------------
# Catalog topology and calibration constants
# ---------------------------------------------------------------------------

AXIS_NAMES: tuple[str, ...] = ("safety", "quality")
"""Two monitoring axes (A = 2, the minimum the library supports)."""

N_METRICS: int = 25
"""Metrics per axis.  ``M = 25`` shrinks the per-axis score noise to
``1/sqrt(M) = 0.2`` so the +1.5 vs +3 separation in [G2] is not seed-lottery."""

METRIC_NAMES: tuple[tuple[str, ...], ...] = tuple(
    tuple(f"{axis}_{i}" for i in range(N_METRICS)) for axis in AXIS_NAMES
)

WINDOW: int = 250
"""Rolling window ``w``.  Must exceed ``k**2`` so the per-axis layer can fire
(see module docstring); 250 also keeps the spike self-inflation ``s^2/w`` small."""

K_SENSITIVITY: float = 7.0
"""Explicit per-axis sensitivity multiplier (see calibration note; paper default
``chi2_A(0.99)`` would make the per-axis layer inert)."""

ALPHA: float = 0.01
"""Joint false-alarm rate (paper default)."""

MIN_WARM: int = 30
"""Joint detector warm-up ticks."""

BASELINE_TICKS: int = 250
"""Baseline ticks before any anomaly onset.  Equal to ``WINDOW`` so the rolling
window is full of in-control data when an anomaly first appears."""

SHIFT_LEN: int = 3
"""Duration (ticks) of the coordinated / subsequence anomaly windows."""

POINT_SHIFT: float = 20.0
"""DGP-02 point magnitude: +20 x the H0 metric std on one axis."""

COORD_SHIFT: float = 1.5
"""DGP-03 coordinated magnitude: +1.5 x the H0 metric std on every axis."""

SUBSEQ_SHIFT: float = 3.0
"""DGP-05 subsequence magnitude: +3 x the H0 metric std on one axis."""

RAMP_RATE: float = 0.05
"""DGP-04 ramp slope: +0.05 x the H0 metric std per tick on one axis."""

CONTAM_FRACTION: float = 0.005
"""DGP-06: fraction of population sessions placed in the winsor clip tail.
0.5% sits *inside* the two-sided p = 0.99 tail (1% per side), so the robust
quantile caps stay stable; 5% would sit above ``q_0.99`` and defeat the cap."""

CONTAM_SHIFT: float = 25.0
"""DGP-06: per-axis shift of contaminated rows, far beyond ``q_0.99``."""

TAXONOMY: dict[str, str] = {
    "DGP-01": "H0 iid baseline",
    "DGP-02": "point",
    "DGP-03": "multivariate-coordinated",
    "DGP-04": "ramp (characterization only)",
    "DGP-05": "subsequence burst",
    "DGP-06": "Tier-2 population contamination",
}
"""DGP id -> Schmidl et al. (2022) anomaly-type label."""


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DGPStream:
    """One full-lifecycle stream emitted by a DGP generator.

    ``ticks`` and ``sessions`` are parallel per-tick sequences; ``sessions[t]``
    may be ``None`` when the DGP emits no Tier-2 rows for that tick (never the
    case for the six catalog DGPs, which always emit at least one session).
    """

    dgp_id: str
    """Catalog id, ``DGP-01`` .. ``DGP-06``."""

    taxonomy: str
    """Schmidl et al. (2022) anomaly-type label."""

    seed: int
    """RNG seed the stream was drawn from."""

    length: int
    """Number of ticks (and session batches)."""

    config: MonitorConfig
    """Validated config the stream conforms to (shapes, axis/metric names)."""

    ticks: tuple[TickMetrics, ...]
    """Per-tick Tier-1 metric matrices (length entries)."""

    sessions: tuple[SessionFeatures | None, ...]
    """Per-tick Tier-2 session batches (length entries)."""

    anomaly_start: int
    """First tick of the Tier-1 anomaly window (``== length`` when none)."""

    anomaly_end: int
    """Exclusive end tick of the Tier-1 anomaly window."""

    layer: str
    """Where the anomaly lives: ``"none"``, ``"tier1"`` or ``"tier2"``."""

    declared: str
    """Human-readable declared expected-flag semantics for the stream."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def default_config() -> MonitorConfig:
    """Return the catalog topology validated with its calibration constants.

    Raises:
        ValueError: propagated from :func:`validate_monitor_config` if a
            calibration constant violates an invariant (it cannot).
    """
    return validate_monitor_config(
        MonitorConfig(
            axis_names=AXIS_NAMES,
            metric_names=METRIC_NAMES,
            window=WINDOW,
            lam=0.25,
            k=K_SENSITIVITY,
            alpha=ALPHA,
            min_warm=MIN_WARM,
            ridge=1e-6,
            agg="mean",
            population_window=80,
            min_population=30,
            winsor_pct=0.99,
            winsor_k=3,
        )
    )


def _baseline_tick(rng: np.random.Generator, cfg: MonitorConfig) -> TickMetrics:
    """One tick of iid N(0, 1) metrics, one array of ``len(names)`` per axis."""
    return TickMetrics(
        X=tuple(
            rng.normal(0.0, 1.0, size=len(names)).astype(np.float64) for names in cfg.metric_names
        )
    )


def _baseline_sessions(
    rng: np.random.Generator,
    cfg: MonitorConfig,
    tick: int,
    n: int,
    prefix: str,
) -> SessionFeatures:
    """``n`` iid N(0, 1)^A session feature rows with unique ids for one tick."""
    a = len(cfg.axis_names)
    return SessionFeatures(
        ids=tuple(f"{prefix}_{tick}_{i}" for i in range(n)),
        tick=tick,
        F=rng.normal(0.0, 1.0, size=(n, a)).astype(np.float64),
    )


def _build(
    dgp_id: str,
    seed: int,
    length: int,
    config: MonitorConfig,
    *,
    shift: Callable[[np.ndarray, int, int], np.ndarray],
    anomaly_start: int,
    anomaly_end: int,
    layer: str,
    declared: str,
    sessions_per_tick: int = 1,
) -> DGPStream:
    """Assemble a stream by applying ``shift`` to each axis's metric array.

    ``shift`` receives ``(metrics, axis_index, tick)`` and returns the
    (possibly shifted) metric array; the identity function yields a pure H0
    baseline.
    """
    rng = np.random.default_rng(seed)
    ticks: list[TickMetrics] = []
    sessions: list[SessionFeatures | None] = []
    for t in range(length):
        X = tuple(
            shift(rng.normal(0.0, 1.0, size=len(names)).astype(np.float64), a, t)
            for a, names in enumerate(config.metric_names)
        )
        ticks.append(TickMetrics(X=X))
        sessions.append(_baseline_sessions(rng, config, t, sessions_per_tick, dgp_id))

    return DGPStream(
        dgp_id=dgp_id,
        taxonomy=TAXONOMY[dgp_id],
        seed=seed,
        length=length,
        config=config,
        ticks=tuple(ticks),
        sessions=tuple(sessions),
        anomaly_start=anomaly_start,
        anomaly_end=anomaly_end,
        layer=layer,
        declared=declared,
    )


def _identity(X: np.ndarray, a: int, t: int) -> np.ndarray:
    """Reference shift: leave the metric array unchanged."""
    return X


# ---------------------------------------------------------------------------
# The six DGP generators (DGP-01 .. DGP-06)
# ---------------------------------------------------------------------------


def dgp_01(seed: int, length: int, *, sessions_per_tick: int = 1) -> DGPStream:
    """DGP-01: H0 iid baseline (no anomaly).

    All metrics and session features are iid N(0, 1).  Declared semantics:
    after warm-up the Tier 1 joint flag rate lies in ``[0.5*alpha, 2*alpha]``
    and the Tier 2 flag rate in ``[alpha, 2*alpha]`` ([P1]/[P3]).

    Args:
        seed: RNG seed.
        length: Number of ticks.
        sessions_per_tick: H0 session rows emitted per tick.

    Returns:
        An H0-only ``DGPStream``.
    """
    return _build(
        "DGP-01",
        seed,
        length,
        default_config(),
        shift=_identity,
        anomaly_start=length,
        anomaly_end=length,
        layer="none",
        declared=(
            "H0 iid: no anomaly; post-warm-up Tier 1 flag rate in [0.5*alpha, "
            "2*alpha] and Tier 2 flag rate in [alpha, 2*alpha]."
        ),
        sessions_per_tick=sessions_per_tick,
    )


def _point_shift(X: np.ndarray, a: int, t: int) -> np.ndarray:
    """DGP-02: +20 sigma on axis 0 for exactly one tick."""
    return X + POINT_SHIFT if (a == 0 and t == BASELINE_TICKS) else X


def dgp_02(seed: int, length: int, *, sessions_per_tick: int = 1) -> DGPStream:
    """DGP-02: point anomaly — single-axis x20-sigma spike for one tick.

    Declared semantics [G2]: at the onset tick ``any_axis_flag`` and
    ``joint_flag`` are both True; recovery is silent as EWMA re-absorbs.

    Args:
        seed: RNG seed.
        length: Number of ticks (must exceed ``BASELINE_TICKS`` to contain the
            anomaly; shorter streams are pure baseline).
        sessions_per_tick: H0 session rows emitted per tick.

    Returns:
        A point-anomaly ``DGPStream``.
    """
    onset = BASELINE_TICKS if length > BASELINE_TICKS else length
    return _build(
        "DGP-02",
        seed,
        length,
        default_config(),
        shift=_point_shift,
        anomaly_start=onset,
        anomaly_end=min(onset + 1, length),
        layer="tier1",
        declared=(
            "point (x20 sigma, 1 tick): any_axis_flag and joint_flag at onset; silent recovery."
        ),
        sessions_per_tick=sessions_per_tick,
    )


def _coord_shift(X: np.ndarray, a: int, t: int) -> np.ndarray:
    """DGP-03: +1.5 sigma on every axis for three ticks."""
    if BASELINE_TICKS <= t < BASELINE_TICKS + SHIFT_LEN:
        return X + COORD_SHIFT
    return X


def dgp_03(seed: int, length: int, *, sessions_per_tick: int = 1) -> DGPStream:
    """DGP-03: multivariate-coordinated — every axis +1.5 sigma for 3 ticks.

    Declared semantics [G2]: during the shift window ``joint_flag`` is True and
    ``any_axis_flag`` is False (each marginal stays within ``k``); the joint
    layer alone fires because the axes move together.

    Args:
        seed: RNG seed.
        length: Number of ticks (``> BASELINE_TICKS + SHIFT_LEN`` to contain the
            full window; shorter streams are warm-up-only).
        sessions_per_tick: H0 session rows emitted per tick.

    Returns:
        A coordinated-shift ``DGPStream``.
    """
    start = BASELINE_TICKS if length > BASELINE_TICKS else length
    end = min(BASELINE_TICKS + SHIFT_LEN, length)
    return _build(
        "DGP-03",
        seed,
        length,
        default_config(),
        shift=_coord_shift,
        anomaly_start=start,
        anomaly_end=end,
        layer="tier1",
        declared=(
            "multivariate-coordinated (+1.5 sigma, 3 ticks): joint_flag and not "
            "any_axis_flag during the shift; each marginal within k."
        ),
        sessions_per_tick=sessions_per_tick,
    )


def _ramp_shift(X: np.ndarray, a: int, t: int) -> np.ndarray:
    """DGP-04: +0.05 sigma/tick on axis 0 from the ramp onset onward."""
    if a == 0 and t >= BASELINE_TICKS:
        return X + RAMP_RATE * (t - BASELINE_TICKS + 1)
    return X


def dgp_04(seed: int, length: int, *, sessions_per_tick: int = 1) -> DGPStream:
    """DGP-04: single-axis ramp (+0.05 sigma/tick) — characterization only.

    Declared semantics [G3]: no flag assertion is made.  The ramp surfaces as a
    near-constant axis-score offset of ``RAMP_RATE * (WINDOW - 1) / 2 ≈ 1.97``
    which the EWMA baseline ``theta`` absorbs (risk R-04); tests assert the
    absorption, not the flags.

    Args:
        seed: RNG seed.
        length: Number of ticks (``> BASELINE_TICKS`` to include drift).
        sessions_per_tick: H0 session rows emitted per tick.

    Returns:
        A ramp ``DGPStream``.
    """
    start = BASELINE_TICKS if length > BASELINE_TICKS else length
    return _build(
        "DGP-04",
        seed,
        length,
        default_config(),
        shift=_ramp_shift,
        anomaly_start=start,
        anomaly_end=length,
        layer="tier1",
        declared=(
            "ramp (+0.05 sigma/tick): characterization only; EWMA theta absorbs "
            "the drift (R-04), no flag assertion."
        ),
        sessions_per_tick=sessions_per_tick,
    )


def _subseq_shift(X: np.ndarray, a: int, t: int) -> np.ndarray:
    """DGP-05: +3 sigma on axis 0 for three consecutive ticks."""
    if a == 0 and BASELINE_TICKS <= t < BASELINE_TICKS + SHIFT_LEN:
        return X + SUBSEQ_SHIFT
    return X


def dgp_05(seed: int, length: int, *, sessions_per_tick: int = 1) -> DGPStream:
    """DGP-05: subsequence burst — single-axis +3 sigma for 3 ticks.

    Declared semantics [G2]: ``any_axis_flag`` and ``joint_flag`` fire at onset;
    recovery is silent as EWMA re-absorbs after the burst.

    Args:
        seed: RNG seed.
        length: Number of ticks (``> BASELINE_TICKS + SHIFT_LEN`` to contain the
            full burst; shorter streams are warm-up-only).
        sessions_per_tick: H0 session rows emitted per tick.

    Returns:
        A subsequence-burst ``DGPStream``.
    """
    start = BASELINE_TICKS if length > BASELINE_TICKS else length
    end = min(BASELINE_TICKS + SHIFT_LEN, length)
    return _build(
        "DGP-05",
        seed,
        length,
        default_config(),
        shift=_subseq_shift,
        anomaly_start=start,
        anomaly_end=end,
        layer="tier1",
        declared=(
            "subsequence burst (+3 sigma, 3 ticks): any_axis_flag and "
            "joint_flag at onset; silent recovery."
        ),
        sessions_per_tick=sessions_per_tick,
    )


def dgp_06(
    seed: int,
    length: int,
    *,
    sessions_per_tick: int = 20,
    contam_fraction: float = CONTAM_FRACTION,
    contam_shift: float = CONTAM_SHIFT,
) -> DGPStream:
    """DGP-06: Tier-2 population contamination.

    The tick stream is H0; a fixed fraction (default 0.5%) of the *session*
    rows are shifted by ``contam_shift`` on axis 0, landing beyond the two-sided
    ``p = 0.99`` winsor clip tail.  Declared semantics [G2]: the winsor caps
    remain stable versus a clean baseline, candidates are never clipped ([P4]),
    and a genuinely shifted candidate is not masked by the contamination.

    Args:
        seed: RNG seed.
        length: Number of ticks (session batches).
        sessions_per_tick: Session rows emitted per tick; ``length *
            sessions_per_tick * contam_fraction`` rows are contaminated.
        contam_fraction: Fraction of rows placed in the winsor tail (0.0 yields
            a clean baseline population for cap-comparison tests).
        contam_shift: Per-axis shift applied to contaminated rows on axis 0.

    Returns:
        A Tier-2 contamination ``DGPStream`` whose contaminated rows carry the
        ``"contam"`` id prefix and clean rows the ``"DGP-06"`` prefix.
    """
    if not 0.0 <= contam_fraction <= 1.0:
        raise ValueError(f"contam_fraction must be in [0, 1]; got {contam_fraction}")

    config = default_config()
    rng = np.random.default_rng(seed)
    ticks = tuple(
        TickMetrics(
            X=tuple(
                rng.normal(0.0, 1.0, size=len(names)).astype(np.float64)
                for names in config.metric_names
            )
        )
        for _ in range(length)
    )

    total = length * sessions_per_tick
    n_contam = max(1, round(total * contam_fraction)) if contam_fraction > 0.0 else 0
    a = len(config.axis_names)
    features = rng.normal(0.0, 1.0, size=(total, a))
    contam_rows = np.unique(np.linspace(0, total - 1, n_contam).round().astype(np.int64))
    features[contam_rows, 0] += contam_shift

    contam_set = {int(i) for i in contam_rows}
    sessions: list[SessionFeatures | None] = []
    offset = 0
    for t in range(length):
        chunk = features[offset : offset + sessions_per_tick]
        ids = tuple(
            f"{'contam' if (offset + i) in contam_set else 'DGP-06'}_{t}_{i}"
            for i in range(sessions_per_tick)
        )
        sessions.append(SessionFeatures(ids=ids, tick=t, F=chunk))
        offset += sessions_per_tick

    return DGPStream(
        dgp_id="DGP-06",
        taxonomy=TAXONOMY["DGP-06"],
        seed=seed,
        length=length,
        config=config,
        ticks=tuple(ticks),
        sessions=tuple(sessions),
        anomaly_start=length,
        anomaly_end=length,
        layer="tier2",
        declared=(
            f"Tier-2 contamination ({contam_fraction:.1%} rows, +{contam_shift} "
            "sigma on axis 0): winsor caps stable vs clean baseline, [P4] "
            "ordering preserved, shifted candidate not masked."
        ),
    )


# ---------------------------------------------------------------------------
# Registry / dispatch
# ---------------------------------------------------------------------------

DGP_IDS: tuple[str, ...] = ("DGP-01", "DGP-02", "DGP-03", "DGP-04", "DGP-05", "DGP-06")
"""Ordered catalog ids."""

DGP_GENERATORS: dict[str, Callable[..., DGPStream]] = {
    "DGP-01": dgp_01,
    "DGP-02": dgp_02,
    "DGP-03": dgp_03,
    "DGP-04": dgp_04,
    "DGP-05": dgp_05,
    "DGP-06": dgp_06,
}
"""Registry of seeded pure generators, keyed by catalog id."""


def generate(dgp_id: str, seed: int, length: int, **kwargs: int | float) -> DGPStream:
    """Dispatch to the named DGP generator.

    Args:
        dgp_id: Catalog id in ``DGP_IDS``.
        seed: RNG seed.
        length: Number of ticks.
        **kwargs: Forwarded generator keyword arguments (e.g.
            ``sessions_per_tick``, ``contam_fraction``).

    Returns:
        The generated ``DGPStream``.

    Raises:
        ValueError: if ``dgp_id`` is not a catalog id.
    """
    generator = DGP_GENERATORS.get(dgp_id)
    if generator is None:
        raise ValueError(f"Unknown dgp_id {dgp_id!r}; choose from {DGP_IDS}")
    return generator(seed, length, **kwargs)
