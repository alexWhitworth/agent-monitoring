# ruff: noqa: N806, N803
"""Tests for joint.py — Welford update, Mahalanobis distance, joint detector."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import chi2

from agent_monitor.config import validate_monitor_config
from agent_monitor.joint import d_squared, update_joint, welford_update
from agent_monitor.types import MonitorConfig, MonitorState, SessionWindow


def _make_fresh_state(
    axis_names: tuple[str, ...] = ("Safety", "Quality", "Efficiency"),
    metric_names: tuple[tuple[str, ...], ...] = (("m1",), ("m1",), ("m1",)),
    min_warm: int = 10,
    alpha: float = 0.01,
    ridge: float = 1e-6,
) -> MonitorState:
    cfg = validate_monitor_config(
        MonitorConfig(
            axis_names=axis_names,
            metric_names=metric_names,
            min_warm=min_warm,
            alpha=alpha,
            ridge=ridge,
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


# ---------------------------------------------------------------------------
# Unit: welford_update
# ---------------------------------------------------------------------------


class TestWelfordUpdate:
    def test_single_update(self) -> None:
        n, mu, M2 = welford_update(0, np.zeros(3), np.zeros((3, 3)), np.array([1.0, 2.0, 3.0]))
        assert n == 1
        np.testing.assert_array_equal(mu, np.array([1.0, 2.0, 3.0]))
        # M2 after 1 sample is all zeros (no scatter yet)
        np.testing.assert_array_equal(M2, np.zeros((3, 3)))

    def test_two_updates(self) -> None:
        n, mu, M2 = 0, np.zeros(2), np.zeros((2, 2))
        n, mu, M2 = welford_update(n, mu, M2, np.array([1.0, 0.0]))
        n, mu, M2 = welford_update(n, mu, M2, np.array([3.0, 0.0]))
        assert n == 2
        np.testing.assert_array_equal(mu, np.array([2.0, 0.0]))
        # scatter = sum (x_i - mu_{i-1})(x_i - mu_i)^T
        # x1=[1,0]; after x1: mu1=[1,0], M2=[[0,0],[0,0]]
        # x2=[3,0]; d_old=[3,0]-[1,0]=[2,0]; mu2=[2,0]
        # M2 = [[0,0],[0,0]] + outer([2,0], [3,0]-[2,0]) = outer([2,0],[1,0]) = [[2,0],[0,0]]
        np.testing.assert_array_equal(M2, np.array([[2.0, 0.0], [0.0, 0.0]]))

    def test_welford_matches_numpy_oracle(self) -> None:
        """Welford mean + scatter matches np.mean/np.cov on the same data."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=(50, 3))
        n, mu, M2 = 0, np.zeros(3), np.zeros((3, 3))
        for row in data:
            n, mu, M2 = welford_update(n, mu, M2, row)
        # Oracle
        expected_mu = np.mean(data, axis=0)
        expected_cov = np.cov(data, rowvar=False, ddof=1)
        expected_scatter = expected_cov * (n - 1)
        np.testing.assert_allclose(mu, expected_mu, atol=1e-12)
        np.testing.assert_allclose(M2, expected_scatter, atol=1e-12)


# ---------------------------------------------------------------------------
# Unit: d_squared
# ---------------------------------------------------------------------------


class TestDSquared:
    def test_scalar_d2(self) -> None:
        S = np.array([2.0, 0.0])
        mu = np.array([0.0, 0.0])
        sigma_inv = np.eye(2)
        assert d_squared(S, mu, sigma_inv) == pytest.approx(4.0)

    def test_batch_d2(self) -> None:
        S = np.array([[2.0, 0.0], [0.0, 3.0]])
        mu = np.array([0.0, 0.0])
        sigma_inv = np.eye(2)
        result = d_squared(S, mu, sigma_inv)
        np.testing.assert_allclose(result, np.array([4.0, 9.0]), atol=1e-12)

    def test_d2_non_negative(self) -> None:
        """D2 >= 0 always (covariance matrix is PSD)."""
        rng = np.random.default_rng(1)
        for _ in range(20):
            A = 5
            S = rng.normal(0, 1, size=A)
            mu = rng.normal(0, 1, size=A)
            # Generate a random PSD matrix for sigma_inv
            Q = rng.normal(0, 1, size=(A, A))
            sigma = Q @ Q.T + 0.1 * np.eye(A)
            sigma_inv = np.linalg.inv(sigma)
            d2 = d_squared(S, mu, sigma_inv)
            assert isinstance(d2, float) and d2 >= -1e-10


# ---------------------------------------------------------------------------
# Unit: update_joint
# ---------------------------------------------------------------------------


class TestUpdateJoint:
    def test_warm_up_returns_none_d2(self) -> None:
        state = _make_fresh_state(min_warm=30)
        S = np.array([0.0, 0.0, 0.0])
        _state_new, result = update_joint(state, S)
        assert result.D2 is None
        assert not result.flagged
        assert "warming up" in result.note
        assert result.n == 1

    def test_after_warm_up_has_d2(self) -> None:
        state = _make_fresh_state(min_warm=5)
        rng = np.random.default_rng(7)
        S = np.zeros(3)
        for _ in range(4):
            S = rng.normal(0, 1, size=3)
            state, _ = update_joint(state, S)
        S = rng.normal(0, 1, size=3)
        state, result = update_joint(state, S)
        assert result.D2 is not None
        assert result.p_value is not None
        assert result.D2 >= 0  # type: ignore[operator]
        assert 0 <= result.p_value <= 1  # type: ignore[operator]

    def test_flagged_equals_predicate(self) -> None:
        """flagged == (D2 > threshold) exactly."""
        state = _make_fresh_state(min_warm=5, alpha=0.01)
        A = 3
        threshold = float(chi2.ppf(0.99, df=A))
        rng = np.random.default_rng(42)
        S = np.zeros(A)
        for _ in range(50):
            S = rng.normal(0, 1, size=A)
            state, result = update_joint(state, S)
            if result.D2 is not None:
                assert result.flagged == (result.D2 > threshold)

    def test_threshold_is_chi2_ppf(self) -> None:
        state = _make_fresh_state(alpha=0.05)
        A = 3
        state, result = update_joint(state, np.array([1.0, 0.0, 0.0]))
        expected = float(chi2.ppf(0.95, df=A))
        assert result.threshold == pytest.approx(expected)

    def test_singular_covariance_raises(self) -> None:
        """Ridge too small with degenerate input raises LinAlgError."""
        state = _make_fresh_state(
            axis_names=("a", "b"),
            metric_names=(("m",), ("m",)),
            min_warm=3,
            ridge=0.0,  # no regularization
        )
        # Feed 2 identical vectors (n=2 < min_warm=3, warm-up only)
        for _ in range(2):
            state, _ = update_joint(state, np.array([0.0, 0.0]))
        # 3rd tick: n=3 >= min_warm, covariance is singular -> LinAlgError
        with pytest.raises(np.linalg.LinAlgError):
            state, _ = update_joint(state, np.array([0.0, 0.0]))


# ---------------------------------------------------------------------------
# Hypothesis: Welford oracle
# ---------------------------------------------------------------------------


_A_strat = st.integers(min_value=2, max_value=6)


class TestHypothesisWelford:
    @given(
        st.integers(min_value=2, max_value=5).flatmap(
            lambda A: st.lists(
                st.lists(
                    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
                    min_size=A,
                    max_size=A,
                ),
                min_size=2,
                max_size=100,
            )
        )
    )
    @settings(max_examples=200)
    def test_welford_vs_numpy(self, data: list[list[float]]) -> None:
        """Welford online matches np.mean/np.cov(ddof=1) on the same sequence."""
        if not data:
            return
        A = len(data[0])
        data_arr = np.array(data)
        n, mu, M2 = 0, np.zeros(A), np.zeros((A, A))
        for row in data_arr:
            n, mu, M2 = welford_update(n, mu, M2, row)
        if n >= 2:
            expected_scatter = np.cov(data_arr, rowvar=False, ddof=1) * (n - 1)
            np.testing.assert_allclose(mu, np.mean(data_arr, axis=0), atol=1e-9)
            np.testing.assert_allclose(M2, expected_scatter, atol=1e-6)


# ---------------------------------------------------------------------------
# Hypothesis: D2 / p_value validity
# ---------------------------------------------------------------------------


class TestHypothesisJoint:
    @given(
        st.lists(
            st.lists(
                st.floats(
                    allow_nan=False, allow_infinity=False,
                    min_value=-1000.0, max_value=1000.0,
                ),
                min_size=3,
                max_size=3,
            ),
            min_size=20,
            max_size=100,
        )
    )
    @settings(max_examples=200)
    def test_d2_non_negative_and_p_value_valid(self, data: list[list[float]]) -> None:
        """D2 >= 0 and p_value in [0, 1] after warm-up."""
        state = _make_fresh_state(min_warm=10)
        for row in data:
            state, result = update_joint(state, np.array(row))
        if result.D2 is not None:
            assert result.D2 >= 0
            assert 0.0 <= result.p_value <= 1.0  # type: ignore[operator]

    @given(
        st.lists(
            st.lists(
                st.floats(
                    allow_nan=False, allow_infinity=False,
                    min_value=-1000.0, max_value=1000.0,
                ),
                min_size=3,
                max_size=3,
            ),
            min_size=20,
            max_size=100,
        )
    )
    @settings(max_examples=200)
    def test_flagged_predicate_exact(self, data: list[list[float]]) -> None:
        """flagged == (D2 > threshold) on all post-warm-up results."""
        state = _make_fresh_state(min_warm=10)
        for row in data:
            state, result = update_joint(state, np.array(row))
            if result.D2 is not None:
                assert result.flagged == (result.D2 > result.threshold)
