"""F-014 tests for the anomaly-type DGP catalog (``tests/dgps.py``).

Covers, per DGP:
  [G1] determinism: same (dgp_id, seed, length) -> byte-identical; differing
       seeds differ (hypothesis).
  [G2] declared expected-flag semantics: point / subsequence fire both layers
       at onset; coordinated fires the joint layer only; Tier-2 contamination
       keeps winsor caps stable and does not mask a shifted candidate.
  [G3] ramp is characterization-only: EWMA absorption asserted, no flag claims.
  [P1]/[P3] DGP-01 H0 flag-rate bands across >= 10 seeds.
"""

from __future__ import annotations

import dgps
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import chi2  # type: ignore[import-untyped]

from agent_monitor.advance import advance
from agent_monitor.population import absorb, score_sessions, winsor_caps
from agent_monitor.types import (
    MonitorConfig,
    MonitorState,
    SessionFeatures,
    SessionWindow,
    TickResult,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

#: Fixed seed verified to realize every [G2] declared-flag semantics.  Chosen
#: because (with the catalog calibration) the +3-sigma subsequence crosses the
#: per-axis threshold at onset while the +1.5-sigma coordinated shift never
#: does; the identity of the seed is immaterial — any catalog seed gives the
#: same baseline distribution, only the realized onset flags move.
SEMANTICS_SEED = 1


def _fresh_state(config: MonitorConfig) -> MonitorState:
    A = len(config.axis_names)
    M_max = max(len(m) for m in config.metric_names)
    w = config.window
    pop = SessionWindow(
        ids=np.array([], dtype=object),
        ticks=np.array([], dtype=np.int64),
        features=np.empty((0, A), dtype=np.float64),
        tick_flagged=np.array([], dtype=bool),
        absorbed_at=np.array([], dtype=np.int64),
    )
    return MonitorState(
        config=config,
        tick=0,
        metric_buf=np.zeros((A, M_max, w), dtype=np.float64),
        metric_count=np.zeros((A, M_max), dtype=np.int64),
        theta=None,
        axis_buf=np.zeros((A, w), dtype=np.float64),
        axis_count=np.zeros(A, dtype=np.int64),
        n=0,
        mu=np.zeros(A, dtype=np.float64),
        M2=np.zeros((A, A), dtype=np.float64),
        population=pop,
    )


def _run_tier1(stream: dgps.DGPStream) -> tuple[MonitorState, list[TickResult]]:
    """Advance a fresh monitor through every tick of ``stream``."""
    state = _fresh_state(stream.config)
    results: list[TickResult] = []
    for tick in stream.ticks:
        state, result = advance(state, tick)
        results.append(result)
    return state, results


def _absorb_all(stream: dgps.DGPStream) -> MonitorState:
    """Absorb every session batch of ``stream`` into a fresh state."""
    state = _fresh_state(stream.config)
    for sf in stream.sessions:
        if sf is not None:
            state = absorb(state, sf, tick_flagged=False)
    return state


def _same_stream(a: dgps.DGPStream, b: dgps.DGPStream) -> bool:
    """Byte-level equality of two streams (ids, features and metric arrays)."""
    if a.dgp_id != b.dgp_id or a.length != b.length or a.seed != b.seed:
        return False
    for ta, tb in zip(a.ticks, b.ticks, strict=True):
        for xa, xb in zip(ta.X, tb.X, strict=True):
            if xa.dtype != xb.dtype or not np.array_equal(xa, xb):
                return False
    for sa, sb in zip(a.sessions, b.sessions, strict=True):
        if (sa is None) != (sb is None):
            return False
        if sa is not None and sb is not None:
            if sa.tick != sb.tick or sa.ids != sb.ids or not np.array_equal(sa.F, sb.F):
                return False
    return True


def _collect_finite(tick: TickResult) -> bool:
    """True when every numeric result field of one tick is finite."""
    fields: list[float | None] = [tick.joint.D2, tick.joint.p_value, tick.joint.threshold]
    for ar in tick.axis_results:
        fields.extend([ar.S_A, ar.theta, ar.sigma_S, ar.deviation, ar.threshold])
    return all(f is None or np.isfinite(f) for f in fields)


# ---------------------------------------------------------------------------
# [G1] Determinism
# ---------------------------------------------------------------------------


@given(
    dgp_id=st.sampled_from(dgps.DGP_IDS),
    seed=st.integers(min_value=0, max_value=10_000),
    length=st.integers(min_value=1, max_value=40),
)
def test_g1_determinism_short_streams(dgp_id: str, seed: int, length: int) -> None:
    """[G1] Same (dgp_id, seed, length) -> byte-identical stream (hypothesis)."""
    assert _same_stream(dgps.generate(dgp_id, seed, length), dgps.generate(dgp_id, seed, length))


@pytest.mark.parametrize("dgp_id", dgps.DGP_IDS)
def test_g1_determinism_full_lifecycle_and_seed_sensitivity(dgp_id: str) -> None:
    """[G1] Byte-identical at anomaly-bearing length; differing seeds differ."""
    a = dgps.generate(dgp_id, 0, dgps.BASELINE_TICKS + 10)
    b = dgps.generate(dgp_id, 0, dgps.BASELINE_TICKS + 10)
    c = dgps.generate(dgp_id, 1, dgps.BASELINE_TICKS + 10)
    assert _same_stream(a, b)
    assert not _same_stream(a, c)


# ---------------------------------------------------------------------------
# Structure / plausibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dgp_id", dgps.DGP_IDS)
def test_stream_shapes_finite_and_ids_unique(dgp_id: str) -> None:
    """All values finite; shapes conform; session ids unique per batch."""
    stream = dgps.generate(dgp_id, 0, dgps.BASELINE_TICKS + 10)
    A = len(stream.config.axis_names)
    names = stream.config.metric_names
    for tick in stream.ticks:
        assert len(tick.X) == A
        for arr, mnames in zip(tick.X, names, strict=True):
            assert arr.shape == (len(mnames),)
            assert arr.dtype == np.float64
            assert np.all(np.isfinite(arr))
    for sf in stream.sessions:
        assert sf is not None
        assert sf.F.shape == (len(sf.ids), A)
        assert sf.F.dtype == np.float64
        assert np.all(np.isfinite(sf.F))
        assert len(set(sf.ids)) == len(sf.ids)


@pytest.mark.parametrize("dgp_id", ["DGP-02", "DGP-03", "DGP-05"])
def test_short_streams_are_warmup_only(dgp_id: str) -> None:
    """Streams shorter than the anomaly onset emit no anomaly window."""
    stream = dgps.generate(dgp_id, 0, dgps.BASELINE_TICKS)
    assert stream.anomaly_start == stream.length
    assert stream.anomaly_end == stream.length
    _, results = _run_tier1(stream)
    assert all(_collect_finite(r) for r in results)


# ---------------------------------------------------------------------------
# [G2] Declared expected-flag semantics
# ---------------------------------------------------------------------------


def test_g2_dgp02_point_fires_both_layers_then_recovers() -> None:
    """[G2] Point (x20 sigma, 1 tick): both layers at onset, silent recovery."""
    stream = dgps.dgp_02(SEMANTICS_SEED, dgps.BASELINE_TICKS + 40)
    _, results = _run_tier1(stream)
    onset = stream.anomaly_start

    # Warm-up ticks preceding onset: a loose sanity bound on gross over-flagging
    # (the precise H0 rate is bounded at scale by test_p1_p3_dgp01_h0_flag_bands;
    # a single alpha=0.01 FP in ~220 ticks is expected, not a bug).
    pre_onset = results[stream.config.min_warm : onset]
    assert sum(r.joint_flag for r in pre_onset) <= 0.05 * len(pre_onset)
    assert results[onset].any_axis_flag
    assert results[onset].joint_flag
    for t in range(onset + 1, onset + 6):
        assert not results[t].any_axis_flag
        assert not results[t].joint_flag


def test_g2_dgp02_point_tp_rate_across_seeds() -> None:
    """[G2] Point: onset TP rate (any_axis_flag and joint_flag), pooled over seeds.

    ``SEMANTICS_SEED`` above pins one concrete realization; this pools 30 seeds
    so the declared onset-firing claim is backed by a rate, not a single draw.
    """
    seeds = range(30)
    hits = 0
    for seed in seeds:
        stream = dgps.dgp_02(seed, dgps.BASELINE_TICKS + 5)
        _, results = _run_tier1(stream)
        onset = stream.anomaly_start
        hits += results[onset].any_axis_flag and results[onset].joint_flag
    assert hits / len(seeds) >= 0.95  # a 20-sigma onset should fire almost always


def test_g2_dgp03_coordinated_joint_only() -> None:
    """[G2] Coordinated (+1.5 sigma, 3 ticks): joint fires, marginals within k."""
    stream = dgps.dgp_03(SEMANTICS_SEED, dgps.BASELINE_TICKS + 40)
    _, results = _run_tier1(stream)
    s, e = stream.anomaly_start, stream.anomaly_end

    # Warm-up ticks preceding the shift: loose sanity bound on gross over-flagging
    # (precise H0 rate is bounded at scale by test_p1_p3_dgp01_h0_flag_bands).
    pre_shift = results[stream.config.min_warm : s]
    assert sum(r.joint_flag for r in pre_shift) <= 0.05 * len(pre_shift)
    for t in range(s, e):
        assert results[t].joint_flag
        assert not results[t].any_axis_flag
        for ar in results[t].axis_results:
            assert ar.deviation is not None and ar.threshold is not None
            assert ar.deviation <= ar.threshold + 1e-12


def test_g2_dgp03_any_axis_fp_rate_across_seeds() -> None:
    """[G2] Coordinated: any_axis_flag should almost never fire during the shift,
    pooled over seeds.

    The per-axis layer staying silent is a probabilistic claim -- ``M=25``
    metrics/axis shrinks marginal noise to ``1/sqrt(25)=0.2`` SD but does not
    eliminate it -- so a single seed cannot bound the false-positive rate.
    """
    n = 0
    fp = 0
    for seed in range(30):
        stream = dgps.dgp_03(seed, dgps.BASELINE_TICKS + 5)
        _, results = _run_tier1(stream)
        for t in range(stream.anomaly_start, stream.anomaly_end):
            n += 1
            fp += results[t].any_axis_flag
    assert fp / n <= 0.02


def test_g2_dgp05_subsequence_fires_at_onset_then_recovers() -> None:
    """[G2] Subsequence (+3 sigma, 3 ticks): both layers at onset, then silent."""
    stream = dgps.dgp_05(SEMANTICS_SEED, dgps.BASELINE_TICKS + 40)
    _, results = _run_tier1(stream)
    s, e = stream.anomaly_start, stream.anomaly_end

    # Warm-up ticks preceding the burst: loose sanity bound on gross over-flagging
    # (precise H0 rate is bounded at scale by test_p1_p3_dgp01_h0_flag_bands).
    pre_burst = results[stream.config.min_warm : s]
    assert sum(r.joint_flag for r in pre_burst) <= 0.05 * len(pre_burst)
    assert results[s].any_axis_flag
    assert results[s].joint_flag
    assert all(results[t].joint_flag for t in range(s, e))
    # The per-axis layer only ever fires at onset: theta's lam=0.25 EWMA chases
    # the sustained +3-sigma shift fast enough that deviation drops back under
    # k*sigma_S by the second burst tick (verified empirically across seeds, not
    # an artifact of this one draw) -- so any_axis_flag is False for the rest of
    # the window even though joint_flag keeps firing.
    for t in range(s + 1, e):
        assert not results[t].any_axis_flag
    for t in range(e, e + 6):
        assert not results[t].any_axis_flag
        assert not results[t].joint_flag


def test_g2_dgp05_subsequence_tp_rate_across_seeds() -> None:
    """[G2] Subsequence: onset TP rate (any_axis_flag and joint_flag), pooled."""
    seeds = range(30)
    hits = 0
    for seed in seeds:
        stream = dgps.dgp_05(seed, dgps.BASELINE_TICKS + 5)
        _, results = _run_tier1(stream)
        onset = stream.anomaly_start
        hits += results[onset].any_axis_flag and results[onset].joint_flag
    assert hits / len(seeds) >= 0.95


def _score_candidate(state: MonitorState, f: list[float]) -> float | None:
    return score_sessions(
        state, SessionFeatures(ids=("cand",), tick=0, F=np.array([f], dtype=np.float64))
    )[0].d2


def test_g2_dgp06_winsor_stability_p4_and_no_masking() -> None:
    """[G2] Tier-2 contamination: stable caps, [P4] ordering, no masking."""
    clean = dgps.dgp_06(0, contam_fraction=0.0, length=100, sessions_per_tick=20)
    contam = dgps.dgp_06(0, contam_fraction=dgps.CONTAM_FRACTION, length=100, sessions_per_tick=20)

    state_clean = _absorb_all(clean)
    state_contam = _absorb_all(contam)

    caps_clean = winsor_caps(state_clean)
    caps_contam = winsor_caps(state_contam)
    assert caps_clean is not None and caps_contam is not None

    # Caps stable versus the clean baseline despite 0.5% contamination.
    np.testing.assert_allclose(caps_contam, caps_clean, atol=0.5)

    upper_axis0 = float(caps_contam[1, 0])

    # [P4] A candidate beyond the cap scores strictly greater than one AT the cap.
    at_cap = _score_candidate(state_contam, [upper_axis0, 0.0])
    beyond = _score_candidate(state_contam, [upper_axis0 + 5.0, 0.0])
    assert at_cap is not None and beyond is not None
    assert beyond > at_cap

    # A genuinely shifted candidate is still detected against the contaminated
    # population, and is not degraded versus the clean population.
    shifted_cand = [6.0, 0.0]
    d2_clean = _score_candidate(state_clean, shifted_cand)
    d2_contam = _score_candidate(state_contam, shifted_cand)
    assert d2_clean is not None and d2_contam is not None
    threshold = float(chi2.ppf(1 - dgps.ALPHA, df=len(dgps.AXIS_NAMES)))
    assert d2_contam > threshold
    assert d2_contam >= 0.5 * d2_clean


def test_g2_dgp06_high_contamination_breaks_cap_stability() -> None:
    """Negative control for the winsor-stability claim.

    Contamination above the tail mass the caps are designed to tolerate should
    move the caps materially, proving the 0.5% stability result in
    ``test_g2_dgp06_winsor_stability_p4_and_no_masking`` isn't vacuously
    insensitive to contamination at any level (module docstring: "5% would sit
    above q_0.99 and defeat the cap").
    """
    clean = dgps.dgp_06(0, contam_fraction=0.0, length=100, sessions_per_tick=20)
    heavy = dgps.dgp_06(0, contam_fraction=0.05, length=100, sessions_per_tick=20)
    caps_clean = winsor_caps(_absorb_all(clean))
    caps_heavy = winsor_caps(_absorb_all(heavy))
    assert caps_clean is not None and caps_heavy is not None
    assert np.max(np.abs(caps_heavy - caps_clean)) > 2.0  # well past the 0.5 stability tolerance


def test_g2_dgp06_contamination_fraction_accounted() -> None:
    """0.5% of emitted DGP-06 rows are contaminated; clean build has none."""
    contam = dgps.dgp_06(0, contam_fraction=0.005, length=100, sessions_per_tick=20)
    clean = dgps.dgp_06(0, contam_fraction=0.0, length=100, sessions_per_tick=20)

    def n_contam(stream: dgps.DGPStream) -> int:
        return sum(
            1 for sf in stream.sessions if sf is not None for i in sf.ids if i.startswith("contam")
        )

    assert n_contam(clean) == 0
    total = sum(len(sf.ids) for sf in contam.sessions if sf is not None)
    assert 1 <= n_contam(contam) <= round(total * 0.01) + 1


# ---------------------------------------------------------------------------
# [G3] Ramp characterization (no flag assertion)
# ---------------------------------------------------------------------------


def test_g3_dgp04_ewma_absorbs_ramp() -> None:
    """[G3] Ramp: EWMA theta absorbs the drift; flags intentionally unasserted.

    The +0.05 sigma/tick ramp drives a sustained positive axis-score offset
    (the rolling z-score tracks the *window* mean, and the window std grows
    with the ramp spread, so the score stabilizes near ``≈ +2`` rather than
    diverging).  The EWMA baseline ``theta`` absorbs this offset (risk R-04);
    this test asserts the absorption, never the flags.
    """
    length = dgps.BASELINE_TICKS + 150
    stream = dgps.dgp_04(SEMANTICS_SEED, length)
    _, results = _run_tier1(stream)

    base_theta = results[dgps.BASELINE_TICKS - 1].axis_results[0].theta
    final_theta = results[-1].axis_results[0].theta
    assert base_theta is not None and final_theta is not None
    assert final_theta > base_theta + 1.0
    assert final_theta > 1.5

    # Characterization, not detection: assert finiteness, not flags.
    assert all(_collect_finite(r) for r in results)


# ---------------------------------------------------------------------------
# [P1]/[P3] DGP-01 H0 flag-rate bands across >= 10 seeds
# ---------------------------------------------------------------------------


def test_p1_p3_dgp01_h0_flag_bands() -> None:
    """[P1]/[P3] DGP-01 realizes H0 flag-rate bands over 12 seeds (>= 10^4).

    This realizes the spec's "H0 flag-rate bounds across seeds" strategy as a
    pooled Monte-Carlo rather than a ``@given`` hypothesis test: a 10^4-sample
    flag-rate band must be pooled across seeds (a single seed's ~10^3 ticks is
    too noisy to bound), so hypothesis is reserved for the many-example
    determinism property above.
    """
    seeds = list(range(12))
    ticks_per_seed = 900  # 12 * 900 = 10 800 ticks and sessions (>= 10^4 post-warm-up)

    # --- Tier 1: joint flag rate over >= 10^4 iid ticks ---
    A = len(dgps.AXIS_NAMES)
    post_warm = 0
    tier1_flags = 0
    d2_values: list[float] = []
    for seed in seeds:
        stream = dgps.dgp_01(seed, ticks_per_seed)
        state = _fresh_state(stream.config)
        for t, tick in enumerate(stream.ticks):
            state, result = advance(state, tick)
            if result.joint.D2 is not None:
                d2_values.append(result.joint.D2)
                if t >= stream.config.min_warm:
                    post_warm += 1
                    if result.joint_flag:
                        tier1_flags += 1
    assert post_warm >= 10_000
    tier1_rate = tier1_flags / post_warm
    assert 0.5 * dgps.ALPHA <= tier1_rate <= 2.0 * dgps.ALPHA
    assert A * 0.95 <= np.mean(d2_values) <= A * 1.05

    # --- Tier 2: d2 flag rate over >= 10^4 H0 candidate sessions ---
    pop_stream = dgps.dgp_01(9999, 1, sessions_per_tick=500)
    pop_state = _absorb_all(pop_stream)
    threshold = float(chi2.ppf(1 - dgps.ALPHA, df=A))

    n_candidates = 0
    tier2_flags = 0
    for seed in seeds:
        cand_stream = dgps.dgp_01(seed, 1, sessions_per_tick=900)
        sf = cand_stream.sessions[0]
        assert sf is not None
        for s in score_sessions(pop_state, sf):
            n_candidates += 1
            if s.d2 is not None and s.d2 > threshold:
                tier2_flags += 1
    assert n_candidates >= 10_000
    tier2_rate = tier2_flags / n_candidates
    assert dgps.ALPHA <= tier2_rate <= 2.0 * dgps.ALPHA
