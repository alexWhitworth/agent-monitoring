# ruff: noqa: N806
"""Tests for session scoring (F-008) — batch-vs-loop equivalence, [P4] ordering,
warm-up gating, and NaN/Inf-free outputs.

API-013: score_sessions(state, sessions) -> tuple[SessionScore, ...]
Reuses API-007 (d_squared) and API-012 (population_stats).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import chi2  # type: ignore[import-untyped]

from agent_monitor.config import validate_monitor_config
from agent_monitor.joint import d_squared
from agent_monitor.population import absorb, population_stats, score_sessions, winsor_caps
from agent_monitor.types import MonitorConfig, MonitorState, SessionFeatures, SessionScore, SessionWindow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fresh_state(
    axis_names: tuple[str, ...] = ("Safety", "Quality"),
    metric_names: tuple[tuple[str, ...], ...] = (("m",), ("m",)),
    population_window: int = 80,
    winsor_pct: float | None = 0.99,
    winsor_k: int = 3,
    min_population: int = 5,
    window: int = 80,
    tick: int = 100,
    **overrides: object,
) -> MonitorState:
    cfg = validate_monitor_config(
        MonitorConfig(
            axis_names=axis_names,
            metric_names=metric_names,
            population_window=population_window,
            winsor_pct=winsor_pct,
            winsor_k=winsor_k,
            min_population=min_population,
            window=window,
            **overrides,
        )
    )
    A = len(cfg.axis_names)
    M_max = max(len(m) for m in cfg.metric_names)
    w = cfg.window

    pop = SessionWindow(
        ids=np.array([], dtype=object),
        ticks=np.array([], dtype=np.int64),
        features=np.empty((0, A), dtype=np.float64),
        tick_flagged=np.array([], dtype=bool),
        absorbed_at=np.array([], dtype=np.int64),
    )

    return MonitorState(
        config=cfg,
        tick=tick,
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


def _seed_population(
    state: MonitorState,
    n_rows: int,
    rng: np.random.Generator | None = None,
    mean: float = 0.0,
    std: float = 1.0,
) -> MonitorState:
    """Absorb n_rows synthetic sessions into a fresh state."""
    if rng is None:
        rng = np.random.default_rng(42)
    A = len(state.config.axis_names)
    features = rng.normal(mean, std, size=(n_rows, A))
    sf = SessionFeatures(
        ids=tuple(str(i) for i in range(n_rows)),
        tick=state.tick,
        F=features,
    )
    return absorb(state, sf, tick_flagged=False)


# ---------------------------------------------------------------------------
# Unit: warm-up gating
# ---------------------------------------------------------------------------


class TestWarmUpGating:
    """Scores must be None when population n < min_population."""

    def test_none_when_population_below_min(self) -> None:
        state = _make_fresh_state(min_population=100)
        sf = SessionFeatures(ids=("a", "b"), tick=0, F=np.array([[1.0, 2.0], [3.0, 4.0]]))
        scores = score_sessions(state, sf)
        for s in scores:
            assert s.d2 is None
            assert s.p_value is None

    def test_none_when_population_empty(self) -> None:
        state = _make_fresh_state(min_population=1)
        sf = SessionFeatures(ids=("a",), tick=0, F=np.array([[5.0, -2.0]]))
        scores = score_sessions(state, sf)
        assert scores[0].d2 is None
        assert scores[0].p_value is None

    def test_scores_available_after_warm_up(self) -> None:
        state = _make_fresh_state(min_population=3, winsor_pct=None, tick=0)
        state = _seed_population(state, n_rows=10)
        sf = SessionFeatures(ids=("cand",), tick=0, F=np.array([[1.0, 0.0]]))
        scores = score_sessions(state, sf)
        assert scores[0].d2 is not None
        assert scores[0].p_value is not None
        assert isinstance(scores[0].d2, float)
        assert isinstance(scores[0].p_value, float)

    def test_none_for_large_batch_during_warm_up(self) -> None:
        state = _make_fresh_state(min_population=50)
        sf = SessionFeatures(
            ids=tuple(str(i) for i in range(30)),
            tick=0,
            F=np.random.default_rng(1).normal(0, 1, size=(30, 2)),
        )
        scores = score_sessions(state, sf)
        assert len(scores) == 30
        for s in scores:
            assert s.d2 is None
            assert s.p_value is None


# ---------------------------------------------------------------------------
# Unit: scoring with winsorized population
# ---------------------------------------------------------------------------


class TestScoringWinsorized:
    """Score sessions against winsorized population."""

    def test_d2_non_negative(self) -> None:
        state = _make_fresh_state(min_population=3, winsor_pct=None, tick=0)
        state = _seed_population(state, n_rows=100, rng=np.random.default_rng(99))
        sf = SessionFeatures(ids=("a",), tick=0, F=np.array([[0.0, 0.0]]))
        scores = score_sessions(state, sf)
        assert scores[0].d2 is not None
        assert scores[0].d2 >= 0.0

    def test_p_value_in_range(self) -> None:
        state = _make_fresh_state(min_population=3, winsor_pct=None, tick=0)
        state = _seed_population(state, n_rows=100, rng=np.random.default_rng(99))
        sf = SessionFeatures(ids=("a",), tick=0, F=np.array([[0.0, 0.0]]))
        scores = score_sessions(state, sf)
        assert scores[0].p_value is not None
        assert 0.0 <= scores[0].p_value <= 1.0

    def test_extreme_candidate_scores_higher(self) -> None:
        """A more extreme candidate (further from mu) should score higher d2."""
        state = _make_fresh_state(min_population=3, winsor_pct=None, tick=0)
        rng = np.random.default_rng(42)
        state = _seed_population(state, n_rows=200, rng=rng)
        sf = SessionFeatures(
            ids=("near_mean", "far"),
            tick=0,
            F=np.array([[0.0, 0.0], [10.0, 10.0]]),
        )
        scores = score_sessions(state, sf)
        assert scores[0].d2 is not None and scores[1].d2 is not None
        assert scores[1].d2 > scores[0].d2

    def test_batch_matches_individual_scoring(self) -> None:
        """Batch scoring should match individual calls."""
        state = _make_fresh_state(min_population=3, winsor_pct=None, tick=0)
        rng = np.random.default_rng(77)
        state = _seed_population(state, n_rows=200, rng=rng)
        F = rng.normal(0, 1, size=(5, 2))
        sf_batch = SessionFeatures(
            ids=("a", "b", "c", "d", "e"), tick=0, F=F,
        )
        scores_batch = score_sessions(state, sf_batch)

        # Score individually
        for i in range(5):
            sf_single = SessionFeatures(
                ids=(f"s{i}",), tick=0, F=F[i : i + 1],
            )
            scores_single = score_sessions(state, sf_single)
            assert scores_batch[i].d2 == pytest.approx(scores_single[0].d2, rel=0, abs=1e-12)
            assert scores_batch[i].p_value == pytest.approx(
                scores_single[0].p_value, rel=0, abs=1e-12
            )


# ---------------------------------------------------------------------------
# Unit: [P4] — candidate beyond cap scores strictly greater d2 than at cap
# ---------------------------------------------------------------------------


class TestP4BeyondCap:
    """Candidates scored RAW (never clipped). A candidate beyond a cap
    must score strictly greater d2 than one at the cap."""

    def test_p4_single_axis_beyond_cap(self) -> None:
        state = _make_fresh_state(
            axis_names=("a", "b"),
            metric_names=(("m",), ("m",)),
            min_population=5,
            population_window=200,
            winsor_pct=0.95,
            window=10,
            winsor_k=20,
            tick=0,
        )
        rng = np.random.default_rng(1)
        state = _seed_population(state, n_rows=300, rng=rng)
        caps = winsor_caps(state)
        assert caps is not None

        cap_upper_axis0 = float(caps[1, 0])
        beyond = cap_upper_axis0 * 3.0

        sf = SessionFeatures(
            ids=("at_cap", "beyond"),
            tick=0,
            F=np.array([[cap_upper_axis0, 0.0], [beyond, 0.0]]),
        )
        scores = score_sessions(state, sf)
        assert scores[0].d2 is not None and scores[1].d2 is not None
        assert scores[1].d2 > scores[0].d2, (
            f"beyond ({beyond}) d2={scores[1].d2} should exceed "
            f"at_cap ({cap_upper_axis0}) d2={scores[0].d2}"
        )

    def test_p4_below_lower_cap(self) -> None:
        state = _make_fresh_state(
            axis_names=("a", "b"),
            metric_names=(("m",), ("m",)),
            min_population=5,
            population_window=200,
            winsor_pct=0.95,
            window=10,
            winsor_k=20,
            tick=0,
        )
        rng = np.random.default_rng(1)
        state = _seed_population(state, n_rows=300, rng=rng)
        caps = winsor_caps(state)
        assert caps is not None

        cap_lower_axis0 = float(caps[0, 0])
        below = cap_lower_axis0 - 5.0

        sf = SessionFeatures(
            ids=("at_lower", "below"),
            tick=0,
            F=np.array([[cap_lower_axis0, 0.0], [below, 0.0]]),
        )
        scores = score_sessions(state, sf)
        assert scores[0].d2 is not None and scores[1].d2 is not None
        assert scores[1].d2 > scores[0].d2, (
            f"below ({below}) d2={scores[1].d2} should exceed "
            f"at_lower ({cap_lower_axis0}) d2={scores[0].d2}"
        )

    def test_p4_both_axes_beyond_caps(self) -> None:
        state = _make_fresh_state(
            axis_names=("a", "b"),
            metric_names=(("m",), ("m",)),
            min_population=5,
            population_window=200,
            winsor_pct=0.95,
            window=10,
            winsor_k=20,
            tick=0,
        )
        rng = np.random.default_rng(1)
        state = _seed_population(state, n_rows=300, rng=rng)
        caps = winsor_caps(state)
        assert caps is not None

        upper_a, upper_b = float(caps[1, 0]), float(caps[1, 1])

        sf = SessionFeatures(
            ids=("at_caps", "beyond_both"),
            tick=0,
            F=np.array([[upper_a, upper_b], [upper_a * 3, upper_b * 3]]),
        )
        scores = score_sessions(state, sf)
        assert scores[0].d2 is not None and scores[1].d2 is not None
        assert scores[1].d2 > scores[0].d2


# ---------------------------------------------------------------------------
# Unit: NaN/Inf-free outputs
# ---------------------------------------------------------------------------


class TestNoNaNInf:
    """All emitted d2/p_value must be finite on valid finite inputs."""

    @pytest.mark.parametrize(
        "F",
        [
            np.array([[0.0, 0.0]]),                     # at origin
            np.array([[1e6, 1e6]]),                     # large magnitudes
            np.array([[-1e6, -1e6]]),                   # large negative
            np.array([[1e-10, 1e-10]]),                 # tiny magnitudes
            np.array([[1e6, -1e6]]),                    # mixed sign large
            np.array([[-1e-10, -1e-10]]),               # tiny negative
        ],
    )
    def test_no_nan_inf_on_extreme_inputs(self, F: np.ndarray) -> None:
        state = _make_fresh_state(min_population=3, winsor_pct=None, tick=0)
        state = _seed_population(state, n_rows=100, rng=np.random.default_rng(99))
        sf = SessionFeatures(ids=("x",), tick=0, F=F)
        scores = score_sessions(state, sf)
        assert scores[0].d2 is not None
        assert scores[0].p_value is not None
        assert np.isfinite(scores[0].d2)
        assert np.isfinite(scores[0].p_value)

    def test_many_candidates_no_nan_inf(self) -> None:
        state = _make_fresh_state(min_population=3, winsor_pct=None, tick=0)
        state = _seed_population(state, n_rows=100, rng=np.random.default_rng(42))
        rng = np.random.default_rng(123)
        F = rng.uniform(-1e6, 1e6, size=(100, 2))
        sf = SessionFeatures(ids=tuple(str(i) for i in range(100)), tick=0, F=F)
        scores = score_sessions(state, sf)
        for s in scores:
            assert s.d2 is not None and np.isfinite(s.d2)
            assert s.p_value is not None and np.isfinite(s.p_value)

    def test_zero_variance_population(self) -> None:
        """All-zero population + ridge should not produce NaN/Inf."""
        state = _make_fresh_state(
            min_population=3, winsor_pct=None, ridge=1e-6, tick=0,
        )
        F_pop = np.zeros((50, 2))
        sf_pop = SessionFeatures(ids=tuple(str(i) for i in range(50)), tick=0, F=F_pop)
        state = absorb(state, sf_pop, tick_flagged=False)
        sf_cand = SessionFeatures(ids=("x",), tick=0, F=np.array([[5.0, 0.0]]))
        scores = score_sessions(state, sf_cand)
        assert scores[0].d2 is not None and np.isfinite(scores[0].d2)
        assert scores[0].p_value is not None and np.isfinite(scores[0].p_value)


# ---------------------------------------------------------------------------
# Unit: chi-square p-value consistency
# ---------------------------------------------------------------------------


class TestChiSquareConsistency:
    """Verify that p_values match chi2.sf(d2, A) independently."""

    def test_p_values_match_chi2_sf(self) -> None:
        state = _make_fresh_state(min_population=3, winsor_pct=None, tick=0)
        rng = np.random.default_rng(55)
        state = _seed_population(state, n_rows=200, rng=rng)
        A = 2
        F = rng.normal(0, 2, size=(10, A))
        sf = SessionFeatures(ids=tuple(str(i) for i in range(10)), tick=0, F=F)
        scores = score_sessions(state, sf)

        stats = population_stats(state)
        assert stats is not None
        mu, sigma_inv, _n = stats
        d2_vals = d_squared(F, mu, sigma_inv)

        for i in range(10):
            expected_p = float(chi2.sf(d2_vals[i], df=A))
            assert scores[i].p_value == pytest.approx(expected_p, rel=0, abs=1e-12)


# ---------------------------------------------------------------------------
# Hypothesis: batch-vs-loop equivalence
# ---------------------------------------------------------------------------


class TestHypothesisBatchVsLoop:
    """Batch d2 must match per-session d_squared loop within 1e-12."""

    @given(
        n_pop=st.integers(min_value=30, max_value=200),
        n_candidates=st.integers(min_value=1, max_value=50),
        axis_count=st.integers(min_value=2, max_value=5),
        winsor_pct_val=st.one_of(
            st.none(),
            st.floats(min_value=0.5, max_value=0.99),
        ),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=200)
    def test_batch_equals_loop(
        self,
        n_pop: int,
        n_candidates: int,
        axis_count: int,
        winsor_pct_val: float | None,
        seed: int,
    ) -> None:
        axis_names = tuple(chr(ord("a") + i) for i in range(axis_count))
        metric_names = tuple(("m",) for _ in range(axis_count))

        state = _make_fresh_state(
            axis_names=axis_names,
            metric_names=metric_names,
            min_population=10,
            population_window=n_pop + 10,
            winsor_pct=winsor_pct_val,
            window=80,
            winsor_k=3 if winsor_pct_val is not None else 1,
            tick=0,
        )
        rng = np.random.default_rng(seed)
        state = _seed_population(state, n_rows=n_pop, rng=rng)

        cand_F = rng.normal(0, 1, size=(n_candidates, axis_count))
        sf = SessionFeatures(
            ids=tuple(str(i) for i in range(n_candidates)),
            tick=0,
            F=cand_F,
        )
        scores_batch = score_sessions(state, sf)

        # Per-session loop
        stats = population_stats(state)
        if stats is None:
            for s in scores_batch:
                assert s.d2 is None
                assert s.p_value is None
            return

        mu, sigma_inv, _n = stats
        d2_batch = d_squared(cand_F, mu, sigma_inv)

        for i in range(n_candidates):
            d2_i = d_squared(cand_F[i], mu, sigma_inv)  # cand_F[i] is (A,) -> scalar
            # Batch d2 vs per-session d2
            assert float(d2_batch[i]) == pytest.approx(float(d2_i), rel=0, abs=1e-10)
            assert scores_batch[i].d2 == pytest.approx(float(d2_i), rel=0, abs=1e-10)
            expected_p = float(chi2.sf(float(d2_i), df=axis_count))
            assert scores_batch[i].p_value == pytest.approx(expected_p, rel=0, abs=1e-10)


# ---------------------------------------------------------------------------
# Hypothesis: [P4] ordering property
# ---------------------------------------------------------------------------


class TestHypothesisP4:
    """Candidate beyond cap always scores higher d2 than at cap."""

    @given(
        n_pop=st.integers(min_value=100, max_value=300),
        axis_count=st.integers(min_value=2, max_value=5),
        winsor_pct_val=st.floats(min_value=0.8, max_value=0.99),
        axis_to_test=st.integers(min_value=0, max_value=4),
        seed=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=100)
    def test_p4_beyond_cap_higher_d2(
        self,
        n_pop: int,
        axis_count: int,
        winsor_pct_val: float,
        axis_to_test: int,
        seed: int,
    ) -> None:
        axis_names = tuple(chr(ord("a") + i) for i in range(axis_count))
        metric_names = tuple(("m",) for _ in range(axis_count))

        # Clamp axis_to_test to valid range
        ax = min(axis_to_test, axis_count - 1)

        state = _make_fresh_state(
            axis_names=axis_names,
            metric_names=metric_names,
            min_population=5,
            population_window=n_pop + 10,
            winsor_pct=winsor_pct_val,
            window=80,
            winsor_k=3,
            tick=0,
        )
        rng = np.random.default_rng(seed)
        state = _seed_population(state, n_rows=n_pop, rng=rng)

        caps = winsor_caps(state)
        if caps is None:
            return  # Skip: warm-up not met for this config

        cap_upper = float(caps[1, ax])
        beyond = cap_upper * 5.0 + abs(cap_upper) + 1.0  # ensure strictly greater

        F_at = np.zeros(axis_count)
        F_beyond = np.zeros(axis_count)
        F_at[ax] = cap_upper
        F_beyond[ax] = beyond

        sf = SessionFeatures(
            ids=("at_cap", "beyond"),
            tick=0,
            F=np.array([F_at, F_beyond]),
        )
        scores = score_sessions(state, sf)
        assert scores[0].d2 is not None and scores[1].d2 is not None
        assert scores[1].d2 > scores[0].d2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases per spec: candidate batch larger than population, etc."""

    def test_candidate_batch_larger_than_population(self) -> None:
        """Scoring many candidates against a small population works."""
        state = _make_fresh_state(min_population=3, winsor_pct=None, tick=0)
        state = _seed_population(state, n_rows=10, rng=np.random.default_rng(1))
        rng = np.random.default_rng(99)
        F = rng.normal(0, 1, size=(500, 2))
        sf = SessionFeatures(ids=tuple(str(i) for i in range(500)), tick=0, F=F)
        scores = score_sessions(state, sf)
        assert len(scores) == 500
        for s in scores:
            assert s.d2 is not None
            assert s.p_value is not None
            assert np.isfinite(s.d2)
            assert np.isfinite(s.p_value)

    def test_empty_candidate_batch(self) -> None:
        state = _make_fresh_state(min_population=3, winsor_pct=None, tick=0)
        state = _seed_population(state, n_rows=100)
        sf = SessionFeatures(ids=(), tick=0, F=np.empty((0, 2)))
        scores = score_sessions(state, sf)
        assert len(scores) == 0

    def test_highly_correlated_population(self) -> None:
        """Near-perfect correlation should still produce valid scores."""
        state = _make_fresh_state(min_population=3, winsor_pct=None, tick=0)
        rng = np.random.default_rng(5)
        x = rng.normal(0, 1, size=200)
        F_pop = np.column_stack([x, x * 0.999 + rng.normal(0, 0.001, size=200)])
        sf_pop = SessionFeatures(ids=tuple(str(i) for i in range(200)), tick=0, F=F_pop)
        state = absorb(state, sf_pop, tick_flagged=False)
        sf_cand = SessionFeatures(ids=("x",), tick=0, F=np.array([[3.0, 3.0]]))
        scores = score_sessions(state, sf_cand)
        assert scores[0].d2 is not None and np.isfinite(scores[0].d2)
        assert scores[0].p_value is not None and np.isfinite(scores[0].p_value)

    def test_single_axis_outlier_in_multi_axis_system(self) -> None:
        """A candidate extreme on one axis, normal on others."""
        state = _make_fresh_state(
            axis_names=("a", "b", "c"),
            metric_names=(("m",), ("m",), ("m",)),
            min_population=3,
            winsor_pct=None,
            tick=0,
        )
        rng = np.random.default_rng(42)
        state = _seed_population(state, n_rows=200, rng=rng)
        sf_at_mean = SessionFeatures(
            ids=("near_mean",), tick=0, F=np.array([[0.0, 0.0, 0.0]])
        )
        sf_outlier = SessionFeatures(
            ids=("outlier",), tick=0, F=np.array([[0.0, 0.0, 15.0]])
        )
        scores_mean = score_sessions(state, sf_at_mean)
        scores_outlier = score_sessions(state, sf_outlier)
        assert scores_outlier[0].d2 is not None and scores_mean[0].d2 is not None
        assert scores_outlier[0].d2 > scores_mean[0].d2