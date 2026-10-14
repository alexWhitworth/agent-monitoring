# ruff: noqa: N806
"""Tests for population.py — absorb, winsor_caps, population_stats, score_sessions."""

from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent_monitor.config import validate_monitor_config
from agent_monitor.population import absorb, population_stats, score_sessions, winsor_caps
from agent_monitor.types import MonitorConfig, MonitorState, SessionFeatures, SessionWindow


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


# ---------------------------------------------------------------------------
# absorb
# ---------------------------------------------------------------------------


class TestAbsorb:
    def test_absorb_adds_rows(self) -> None:
        state = _make_fresh_state(population_window=10, winsor_k=1, window=10, tick=0)
        sf = SessionFeatures(
            ids=("a", "b"), tick=0, F=np.array([[1.0, 2.0], [3.0, 4.0]])
        )
        state = absorb(state, sf, tick_flagged=False)
        assert len(state.population.ids) == 2
        assert list(state.population.ids) == ["a", "b"]

    def test_absorb_dedups_existing(self) -> None:
        state = _make_fresh_state(population_window=10, winsor_k=1, window=10, tick=0)
        sf1 = SessionFeatures(ids=("a",), tick=0, F=np.array([[1.0, 2.0]]))
        state = absorb(state, sf1, tick_flagged=False)
        sf2 = SessionFeatures(ids=("a", "b"), tick=0, F=np.array([[1.0, 2.0], [3.0, 4.0]]))
        state = absorb(state, sf2, tick_flagged=False)
        assert len(state.population.ids) == 2  # a already exists, b added
        assert "a" in set(state.population.ids)
        assert "b" in set(state.population.ids)

    def test_absorb_dedups_within_batch(self) -> None:
        state = _make_fresh_state(population_window=10, winsor_k=1, window=10, tick=0)
        sf = SessionFeatures(
            ids=("a", "a", "b"), tick=0, F=np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
        )
        state = absorb(state, sf, tick_flagged=False)
        assert len(state.population.ids) == 2  # dedup within batch

    def test_absorb_evicts_old_rows(self) -> None:
        state = _make_fresh_state(population_window=5, winsor_k=1, window=5, tick=10)
        # Add row at old tick
        sf = SessionFeatures(ids=("old",), tick=4, F=np.array([[1.0, 2.0]]))
        state = absorb(state, sf, tick_flagged=False)
        # retention = max(5, 5*1) = 5, cutoff = 10 - 5 + 1 = 6
        # old tick 4 < 6 should be evicted
        assert len(state.population.ids) == 0

    def test_absorb_records_tick_flagged(self) -> None:
        state = _make_fresh_state(population_window=10, winsor_k=1, window=10, tick=0)
        sf = SessionFeatures(ids=("a",), tick=0, F=np.array([[1.0, 2.0]]))
        state = absorb(state, sf, tick_flagged=True)
        assert state.population.tick_flagged[0]

    def test_absorb_dedup_all_with_eviction(self) -> None:
        """All session ids already retained; eviction still occurs."""
        state = _make_fresh_state(population_window=5, winsor_k=1, window=5, tick=10)
        # Add a row at tick=6 (within retention window: cutoff=10-5+1=6)
        sf1 = SessionFeatures(ids=("keep",), tick=6, F=np.array([[1.0, 2.0]]))
        state = absorb(state, sf1, tick_flagged=False)
        assert len(state.population.ids) == 1
        # Advance tick, then call absorb with same id -> dedup, but eviction still runs
        object.__setattr__(state, "tick", 15)  # cutoff = 15-5+1 = 11
        sf2 = SessionFeatures(ids=("keep",), tick=14, F=np.array([[1.0, 2.0]]))
        state = absorb(state, sf2, tick_flagged=False)
        # The old row at tick=6 should be evicted (6 < 11), no new row added
        assert len(state.population.ids) == 0

    def test_absorb_no_new_rows_no_eviction(self) -> None:
        """All ids duplicates and all ticks recent: fast-path returns state."""
        state = _make_fresh_state(population_window=10, winsor_k=1, window=10, tick=0)
        sf1 = SessionFeatures(ids=("a",), tick=0, F=np.array([[1.0, 2.0]]))
        state = absorb(state, sf1, tick_flagged=False)
        assert len(state.population.ids) == 1
        # Same tick, same id, all recent: no-op
        sf2 = SessionFeatures(ids=("a",), tick=0, F=np.array([[1.0, 2.0]]))
        state2 = absorb(state, sf2, tick_flagged=False)
        assert state2 is state  # fast path returned same state object


# ---------------------------------------------------------------------------
# winsor_caps
# ---------------------------------------------------------------------------


class TestWinsorCaps:
    def test_none_when_winsor_pct_is_none(self) -> None:
        state = _make_fresh_state(winsor_pct=None)
        assert winsor_caps(state) is None

    def test_none_when_insufficient_rows(self) -> None:
        state = _make_fresh_state(winsor_pct=0.99)
        assert winsor_caps(state) is None  # 0 < 100

    def test_caps_computed_from_full_retention(self) -> None:
        state = _make_fresh_state(
            axis_names=("a", "b"), metric_names=(("m",), ("m",)),
            winsor_pct=0.99, window=10, winsor_k=2, tick=0,
        )
        # Add 150 rows with known distribution
        rng = np.random.default_rng(1)
        features = rng.normal(0, 1, size=(150, 2))
        sf = SessionFeatures(
            ids=tuple(str(i) for i in range(150)),
            tick=0,
            F=features,
        )
        state = absorb(state, sf, tick_flagged=False)
        caps = winsor_caps(state)
        assert caps is not None
        assert caps.shape == (2, 2)
        assert caps[0, 0] <= caps[1, 0]  # lower <= upper
        # Should be close to normal quantiles
        assert caps[0, 0] < -2.0  # q_0.01 of N(0,1) ≈ -2.33
        assert caps[1, 0] > 2.0   # q_0.99 of N(0,1) ≈ 2.33


# ---------------------------------------------------------------------------
# population_stats
# ---------------------------------------------------------------------------


class TestPopulationStats:
    def test_none_until_min_population(self) -> None:
        state = _make_fresh_state(min_population=10)
        assert population_stats(state) is None

    def test_stats_after_sufficient_rows(self) -> None:
        state = _make_fresh_state(
            axis_names=("a", "b"), metric_names=(("m",), ("m",)),
            min_population=5, population_window=100, winsor_pct=None, tick=0,
        )
        rng = np.random.default_rng(42)
        features = rng.normal(0, 1, size=(20, 2))
        sf = SessionFeatures(
            ids=tuple(str(i) for i in range(20)),
            tick=0,
            F=features,
        )
        state = absorb(state, sf, tick_flagged=False)
        stats = population_stats(state)
        assert stats is not None
        mu, sigma_inv, n = stats
        assert mu.shape == (2,)
        assert sigma_inv.shape == (2, 2)
        assert n == 20

    def test_stats_with_winsorization(self) -> None:
        state = _make_fresh_state(
            axis_names=("a", "b"), metric_names=(("m",), ("m",)),
            min_population=5, population_window=100, winsor_pct=0.99,
            window=10, winsor_k=10, tick=0,
        )
        rng = np.random.default_rng(99)
        features = rng.normal(0, 1, size=(200, 2))
        sf = SessionFeatures(
            ids=tuple(str(i) for i in range(200)),
            tick=0,
            F=features,
        )
        state = absorb(state, sf, tick_flagged=False)
        stats = population_stats(state)
        assert stats is not None
        mu, sigma_inv, _n = stats
        # With winsorization at 0.99, effective variance should be smaller
        assert mu.shape == (2,)
        assert sigma_inv.shape == (2, 2)

    def test_singular_covariance_raises(self) -> None:
        state = _make_fresh_state(
            axis_names=("a", "b"), metric_names=(("m",), ("m",)),
            min_population=3, population_window=100, winsor_pct=None,
            ridge=0.0, tick=0,
        )
        # Identical rows -> singular covariance
        F = np.zeros((10, 2))
        sf = SessionFeatures(ids=tuple(str(i) for i in range(10)), tick=0, F=F)
        state = absorb(state, sf, tick_flagged=False)
        with pytest.raises(
            np.linalg.LinAlgError,
            match=r"Population ridge-regularized covariance is singular \(ridge=0\.0\)",
        ):
            population_stats(state)


# ---------------------------------------------------------------------------
# score_sessions
# ---------------------------------------------------------------------------


class TestScoreSessions:
    def test_none_scores_during_warm_up(self) -> None:
        state = _make_fresh_state(min_population=50)
        sf = SessionFeatures(ids=("a",), tick=0, F=np.array([[1.0, 2.0]]))
        scores = score_sessions(state, sf)
        assert scores[0].d2 is None
        assert scores[0].p_value is None

    def test_scores_after_warm_up(self) -> None:
        state = _make_fresh_state(
            axis_names=("a", "b"), metric_names=(("m",), ("m",)),
            min_population=5, population_window=100, winsor_pct=None, tick=0,
        )
        rng = np.random.default_rng(7)
        features = rng.normal(0, 1, size=(50, 2))
        sf_pop = SessionFeatures(
            ids=tuple(str(i) for i in range(50)),
            tick=0,
            F=features,
        )
        state = absorb(state, sf_pop, tick_flagged=False)
        # Score a candidate
        sf_cand = SessionFeatures(ids=("cand",), tick=0, F=np.array([[3.0, 0.0]]))
        scores = score_sessions(state, sf_cand)
        assert scores[0].d2 is not None
        assert scores[0].p_value is not None
        assert scores[0].d2 >= 0

    def test_candidates_scored_raw_not_clipped(self) -> None:
        """Candidates never clipped; extreme candidates get higher d2."""
        state = _make_fresh_state(
            axis_names=("a", "b"), metric_names=(("m",), ("m",)),
            min_population=5, population_window=100, winsor_pct=0.99,
            window=10, winsor_k=10, tick=0,
        )
        rng = np.random.default_rng(1)
        features = rng.normal(0, 1, size=(200, 2))
        sf_pop = SessionFeatures(
            ids=tuple(str(i) for i in range(200)),
            tick=0,
            F=features,
        )
        state = absorb(state, sf_pop, tick_flagged=False)
        # Score candidates: one at the cap, one beyond
        caps = winsor_caps(state)
        assert caps is not None
        cap_val = float(caps[1, 0])
        beyond_val = cap_val * 2
        sf_cand = SessionFeatures(
            ids=("at_cap", "beyond"),
            tick=0,
            F=np.array([[cap_val, 0.0], [beyond_val, 0.0]]),
        )
        scores = score_sessions(state, sf_cand)
        assert scores[0].d2 is not None and scores[1].d2 is not None
        # [P4]: candidate beyond cap scores strictly greater d2
        assert scores[1].d2 > scores[0].d2  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Hypothesis: absorb invariants
# ---------------------------------------------------------------------------


class TestHypothesisAbsorb:
    @given(
        st.lists(
            st.tuples(
                st.text(min_size=1, max_size=20),
                st.integers(min_value=0, max_value=100),
                st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000),
                st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000),
            ),
            min_size=1,
            max_size=50,
        ),
        st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=100)
    def test_no_duplicate_ids_after_absorb(
        self, rows: list[tuple[str, int, float, float]], tick: int
    ) -> None:
        state = _make_fresh_state(
            population_window=20, winsor_k=1, window=20, tick=tick,
        )
        ids = [r[0] for r in rows]
        features = np.array([[r[2], r[3]] for r in rows])
        sf = SessionFeatures(ids=tuple(ids), tick=0, F=features)
        state = absorb(state, sf, tick_flagged=False)
        # No duplicates in retained rows
        retained = list(state.population.ids)
        assert len(retained) == len(set(retained))


# ---------------------------------------------------------------------------
# Mutation hardening (F-013): exact boundary, dtype, and state-preservation
# ---------------------------------------------------------------------------


class TestAbsorbHardening:
    def test_retention_is_window_times_winsor_k(self) -> None:
        """retention = max(population_window, window * winsor_k)."""
        # population_window=2 < window * winsor_k = 18
        state = _make_fresh_state(population_window=2, window=6, winsor_k=3, tick=100)
        sf = SessionFeatures(ids=("r",), tick=85, F=np.array([[1.0, 2.0]]))
        state = absorb(state, sf, tick_flagged=False)
        # cutoff = 100 - 18 + 1 = 83; tick 85 >= 83 -> retained
        assert len(state.population.ids) == 1

    def test_evict_only_boundary_row_retained(self) -> None:
        """Evict-only path keeps a row whose tick equals the cutoff exactly."""
        state = _make_fresh_state(population_window=5, window=5, winsor_k=1, tick=10)
        sf1 = SessionFeatures(ids=("edge",), tick=6, F=np.array([[1.0, 2.0]]))
        state = absorb(state, sf1, tick_flagged=False)
        assert len(state.population.ids) == 1
        # Re-absorb the same id: no new rows -> evict-only path; cutoff still 6
        sf2 = SessionFeatures(ids=("edge",), tick=10, F=np.array([[0.0, 0.0]]))
        state = absorb(state, sf2, tick_flagged=False)
        assert list(state.population.ids) == ["edge"]
        assert state.population.ticks[0] == 6

    def test_evict_only_preserves_all_state_fields(self) -> None:
        """Evict-only absorb preserves every Tier 1 field and the retained rows."""
        base = _make_fresh_state(population_window=5, window=5, winsor_k=1, tick=10)
        theta = np.array([1.0, 2.0])
        theta.flags.writeable = False
        mu = np.array([3.0, 4.0])
        mu.flags.writeable = False
        M2 = np.array([[5.0, 1.0], [1.0, 6.0]])
        M2.flags.writeable = False
        state = replace(base, theta=theta, n=3, mu=mu, M2=M2)
        pop = SessionWindow(
            ids=np.array(["old", "keep"], dtype=object),
            ticks=np.array([4, 7], dtype=np.int64),  # cutoff=6: old evicted, keep retained
            features=np.array([[9.0, 9.0], [1.0, 2.0]]),
            tick_flagged=np.array([True, False]),
            absorbed_at=np.array([0, 2], dtype=np.int64),
        )
        state = replace(state, population=pop)
        sf = SessionFeatures(ids=("keep",), tick=10, F=np.array([[0.0, 0.0]]))
        state2 = absorb(state, sf, tick_flagged=False)
        # Tier 1 state preserved
        assert state2.config is state.config
        assert state2.tick == 10
        assert state2.metric_buf is state.metric_buf
        assert state2.metric_count is state.metric_count
        assert state2.theta is state.theta
        assert state2.axis_buf is state.axis_buf
        assert state2.axis_count is state.axis_count
        assert state2.n == 3
        np.testing.assert_array_equal(state2.mu, mu)
        np.testing.assert_array_equal(state2.M2, M2)
        # Population: only the retained row, with all its fields
        assert list(state2.population.ids) == ["keep"]
        assert state2.population.ticks[0] == 7
        np.testing.assert_array_equal(state2.population.features, [[1.0, 2.0]])
        assert not state2.population.tick_flagged[0]
        assert state2.population.absorbed_at[0] == 2

    def test_absorb_result_dtypes(self) -> None:
        """SessionWindow arrays keep their contract dtypes after absorbing new rows."""
        state = _make_fresh_state(population_window=10, window=5, winsor_k=1, tick=0)
        sf = SessionFeatures(ids=("a", "b"), tick=0, F=np.array([[1.0, 2.0], [3.0, 4.0]]))
        state = absorb(state, sf, tick_flagged=True)
        pop = state.population
        assert pop.ids.dtype == object
        assert pop.ticks.dtype == np.int64
        assert pop.features.dtype == np.float64
        assert pop.tick_flagged.dtype == np.bool_
        assert pop.absorbed_at.dtype == np.int64

    def test_absorb_preserves_existing_row_tick_and_absorbed_at(self) -> None:
        """New-row writes must not clobber existing rows' tick/absorbed_at."""
        base = _make_fresh_state(population_window=100, window=5, winsor_k=1, tick=20)
        pop = SessionWindow(
            ids=np.array(["old"], dtype=object),
            ticks=np.array([10], dtype=np.int64),
            features=np.array([[5.0, 5.0]]),
            tick_flagged=np.array([False]),
            absorbed_at=np.array([5], dtype=np.int64),
        )
        state = replace(base, population=pop)
        sf = SessionFeatures(ids=("n1", "n2"), tick=20, F=np.array([[1.0, 0.0], [0.0, 1.0]]))
        state = absorb(state, sf, tick_flagged=False)
        pop2 = state.population
        assert list(pop2.ids) == ["old", "n1", "n2"]
        assert pop2.ticks[0] == 10
        assert pop2.ticks[1] == 20 and pop2.ticks[2] == 20
        assert pop2.absorbed_at[0] == 5
        assert pop2.absorbed_at[1] == 20 and pop2.absorbed_at[2] == 20

    def test_absorb_preserves_theta_with_new_rows(self) -> None:
        base = _make_fresh_state(population_window=10, window=5, winsor_k=1, tick=0)
        theta = np.array([1.0, 2.0])
        theta.flags.writeable = False
        state = replace(base, theta=theta)
        sf = SessionFeatures(ids=("new",), tick=0, F=np.array([[1.0, 2.0]]))
        state2 = absorb(state, sf, tick_flagged=False)
        assert state2.theta is state.theta


class TestWinsorCapsHardening:
    def test_boundary_exactly_100_rows(self) -> None:
        """caps available at exactly 100 retained rows; None at 99."""

        def make(n: int) -> MonitorState:
            state = _make_fresh_state(
                population_window=500, window=5, winsor_k=1, tick=0, winsor_pct=0.9
            )
            F = np.column_stack(
                [np.arange(1, n + 1, dtype=float), np.arange(n, 0, -1, dtype=float)]
            )
            sf = SessionFeatures(ids=tuple(f"s{i}" for i in range(n)), tick=0, F=F)
            return absorb(state, sf, tick_flagged=False)

        assert winsor_caps(make(99)) is None
        assert winsor_caps(make(100)) is not None


class TestPopulationStatsHardening:
    @staticmethod
    def _state_with_rows(
        rows: list[tuple[int, list[float]]],
        **cfg_kwargs: object,
    ) -> MonitorState:
        state = _make_fresh_state(**cfg_kwargs)
        pop = SessionWindow(
            ids=np.array([f"r{i}" for i in range(len(rows))], dtype=object),
            ticks=np.array([r[0] for r in rows], dtype=np.int64),
            features=np.array([r[1] for r in rows], dtype=np.float64),
            tick_flagged=np.zeros(len(rows), dtype=bool),
            absorbed_at=np.zeros(len(rows), dtype=np.int64),
        )
        return replace(state, population=pop)

    def test_cov_window_boundary_inclusive(self) -> None:
        """Rows at tick == cov cutoff are included; older rows are not."""
        # retention = max(10, 10*3) = 30 > population_window = 10
        state = self._state_with_rows(
            [
                (40, [100.0, 0.0]),  # in population (>= 21), outside cov window
                (41, [0.0, 100.0]),  # exactly at cov cutoff 50-10+1=41 -> included
                (45, [0.0, 0.0]),
                (45, [0.0, 0.0]),
                (45, [0.0, 0.0]),
            ],
            population_window=10,
            window=10,
            winsor_k=3,
            tick=50,
            winsor_pct=None,
            min_population=4,
        )
        stats = population_stats(state)
        assert stats is not None
        mu, _sigma_inv, n = stats
        assert n == 4
        np.testing.assert_allclose(mu, [0.0, 25.0])

    def test_min_population_boundary(self) -> None:
        """stats present at exactly n == min_population; None at n - 1."""
        rows = [(10, [1.0, 2.0]), (10, [3.0, 4.0]), (10, [5.0, 6.0]), (10, [7.0, 8.0])]
        state4 = self._state_with_rows(
            rows, population_window=20, window=5, winsor_k=1, tick=10,
            winsor_pct=None, min_population=4,
        )
        assert population_stats(state4) is not None
        state3 = self._state_with_rows(
            rows[:3], population_window=20, window=5, winsor_k=1, tick=10,
            winsor_pct=None, min_population=4,
        )
        assert population_stats(state3) is None

    def test_winsorized_stats_match_independent_oracle(self) -> None:
        """mu/Sigma_inv equal a numpy oracle over the clipped rows, exactly."""
        n_rows = 120
        rng = np.random.default_rng(11)
        F = np.column_stack(
            [np.arange(1, n_rows + 1, dtype=float), rng.normal(0.0, 5.0, size=n_rows)]
        )
        state = _make_fresh_state(
            population_window=200, window=5, winsor_k=1, tick=0,
            winsor_pct=0.75, min_population=5, ridge=0.01,
        )
        sf = SessionFeatures(ids=tuple(f"s{i}" for i in range(n_rows)), tick=0, F=F)
        state = absorb(state, sf, tick_flagged=False)

        stats = population_stats(state)
        assert stats is not None
        mu, sigma_inv, n = stats

        p = 0.75
        lower = np.quantile(F, 1.0 - p, axis=0)
        upper = np.quantile(F, p, axis=0)
        clipped = np.clip(F, lower, upper)
        mu_o = clipped.mean(axis=0)
        Sigma_o = np.cov(clipped, rowvar=False, ddof=1) + 0.01 * np.eye(2)
        inv_o = np.linalg.inv(Sigma_o)

        assert n == n_rows
        np.testing.assert_allclose(mu, mu_o, rtol=1e-12)
        np.testing.assert_allclose(sigma_inv, inv_o, rtol=1e-9)


class TestScoreSessionsHardening:
    def test_ids_preserved_warm_up(self) -> None:
        state = _make_fresh_state(min_population=50)
        sf = SessionFeatures(ids=("x", "y"), tick=0, F=np.array([[1.0, 2.0], [3.0, 4.0]]))
        scores = score_sessions(state, sf)
        assert [s.id for s in scores] == ["x", "y"]
        assert all(s.d2 is None for s in scores)

    def test_ids_preserved_scored(self) -> None:
        state = _make_fresh_state(
            min_population=5, population_window=100, winsor_pct=None, tick=0
        )
        rng = np.random.default_rng(3)
        F = rng.normal(0.0, 1.0, size=(20, 2))
        sf_pop = SessionFeatures(ids=tuple(f"p{i}" for i in range(20)), tick=0, F=F)
        state = absorb(state, sf_pop, tick_flagged=False)
        sf = SessionFeatures(ids=("x", "y"), tick=0, F=np.array([[1.0, 2.0], [3.0, 4.0]]))
        scores = score_sessions(state, sf)
        assert [s.id for s in scores] == ["x", "y"]
        assert all(s.d2 is not None for s in scores)
