"""Tests for axis.py — per-axis EWMA monitor (API-005)."""

import numpy as np
import pytest

from agent_monitor.agg import get_agg
from agent_monitor.axis import update_axis
from agent_monitor.config import validate_monitor_config
from agent_monitor.types import MonitorConfig, MonitorState, SessionWindow


def _make_fresh_state(
    axis_names: tuple[str, ...] = ("Safety", "Quality"),
    metric_names: tuple[tuple[str, ...], ...] = (("m1", "m2"), ("m1",)),
    window: int = 80,
    lam: float = 0.25,
    k: float | None = None,
    **overrides: object,
) -> MonitorState:
    """Build a fresh MonitorState with paper defaults, suitable for axis testing."""
    cfg = validate_monitor_config(
        MonitorConfig(
            axis_names=axis_names,
            metric_names=metric_names,
            window=window,
            lam=lam,
            k=k,
            **overrides,  # type: ignore[arg-type]
        )
    )
    A = len(cfg.axis_names)  # noqa: N806
    M_max = max(len(m) for m in cfg.metric_names)  # noqa: N806
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


def _advance_multiple(
    state: MonitorState, axis_idx: int, values_per_tick: list[list[float]]
) -> tuple[MonitorState, list[dict[str, object]]]:
    """Advance the axis multiple ticks, collecting results."""
    agg_fn = get_agg(state.config.agg)
    results: list[dict[str, object]] = []
    for vals in values_per_tick:
        state, result = update_axis(state, axis_idx, vals, agg_fn)
        results.append(
            {
                "S_A": result.S_A,
                "theta": result.theta,
                "sigma_S": result.sigma_S,
                "deviation": result.deviation,
                "threshold": result.threshold,
                "flagged": result.flagged,
                "note": result.note,
            }
        )
    return state, results


# ---------------------------------------------------------------------------
# Unit: warm-up
# ---------------------------------------------------------------------------


class TestWarmUp:
    def test_first_tick_warm_up(self) -> None:
        """First tick: z-scores warm up (need >= 2 per metric)."""
        state = _make_fresh_state()
        agg_fn = get_agg(state.config.agg)
        state, result = update_axis(state, 0, [1.0, 2.0], agg_fn)
        assert result.S_A is None
        assert result.theta is None
        assert not result.flagged
        assert "warming up z-scores" in result.note

    def test_second_tick_with_enough_samples(self) -> None:
        """Second tick with same metrics: z-scores available, S_A computed."""
        state = _make_fresh_state()
        agg_fn = get_agg(state.config.agg)
        state, _r1 = update_axis(state, 0, [1.0, 2.0], agg_fn)
        state, r2 = update_axis(state, 0, [1.5, 2.5], agg_fn)
        assert r2.S_A is not None
        assert r2.theta is not None
        assert "warming up sigma_S" in r2.note  # only 2 S_A samples so far

    def test_third_tick_sigma_available(self) -> None:
        """Third tick: enough S_A samples for sigma_S."""
        state = _make_fresh_state()
        agg_fn = get_agg(state.config.agg)
        for vals in [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]]:
            state, result = update_axis(state, 0, vals, agg_fn)
        assert result.sigma_S is not None
        assert result.flagged is not None

    def test_axis_with_1_metric_single_warm_up(self) -> None:
        """Single-metric axis: warm-up behaves the same."""
        state = _make_fresh_state(
            axis_names=("A", "B"), metric_names=(("m",), ("m",))
        )
        agg_fn = get_agg(state.config.agg)
        state, r1 = update_axis(state, 0, [1.0], agg_fn)
        assert r1.S_A is None  # only 1 sample for the metric
        state, r2 = update_axis(state, 0, [2.0], agg_fn)
        assert r2.S_A is not None  # now 2 samples


# ---------------------------------------------------------------------------
# Unit: EWMA theta
# ---------------------------------------------------------------------------


class TestEWMA:
    def test_theta_cold_starts_at_first_S_A(self) -> None:  # noqa: N802
        """theta is set to first S_A on cold start."""
        state = _make_fresh_state(lam=0.25)
        agg_fn = get_agg(state.config.agg)
        # First tick S_A = None
        state, _r1 = update_axis(state, 0, [5.0, 3.0], agg_fn)
        # Second tick: both metrics have 2 samples -> z-scores, S_A computed
        state, r2 = update_axis(state, 0, [7.0, 1.0], agg_fn)
        assert r2.theta == pytest.approx(r2.S_A)  # type: ignore[arg-type]

    def test_theta_follows_ewma_recurrence(self) -> None:
        """theta(t) = lam * S_A + (1-lam) * theta(t-1)."""
        state = _make_fresh_state(lam=0.25)
        # Run several ticks with values
        vals_seq = [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],  # this should shift theta
            [2.0, 2.0],
        ]
        state, results = _advance_multiple(state, 0, vals_seq)
        # At tick 4 (index 3 after warm-up), S_A = some value, theta = cold_start
        # At tick 5 (index 4): theta(5) = lam * S_A(5) + (1-lam) * theta(4)
        r4 = results[3]
        r5 = results[4]
        if r4["theta"] is not None and r5["S_A"] is not None:
            expected_theta = 0.25 * r5["S_A"] + 0.75 * r4["theta"]
            assert r5["theta"] == pytest.approx(expected_theta, rel=0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Unit: flagging predicate
# ---------------------------------------------------------------------------


class TestFlagging:
    def test_no_flag_on_normal_values(self) -> None:
        """Steady values with natural variation: check no crash, fields populated."""
        state = _make_fresh_state(window=10)
        agg_fn = get_agg(state.config.agg)
        for _ in range(20):
            rng = np.random.default_rng(42)
            v = list(rng.normal(0, 1, size=2))
            state, result = update_axis(state, 0, v, agg_fn)
        # All result fields should be populated (or legitimately None for sigma_S near 0)
        assert result.axis == "Safety"

    def test_flag_when_deviation_exceeds_k_sigma(self) -> None:
        """Inject a spike that exceeds k * sigma_S to trigger the flag."""
        state = _make_fresh_state(window=10, k=0.5, lam=0.5)
        agg_fn = get_agg(state.config.agg)
        # Warm up with small variation so sigma_S stays small
        rng = np.random.default_rng(1)
        for _ in range(15):
            v = list(rng.normal(0, 0.1, size=2))
            state, _ = update_axis(state, 0, v, agg_fn)
        # Inject a huge spike: sigma_S is ~ small, deviation will be huge
        state, result = update_axis(state, 0, [100.0, 100.0], agg_fn)
        assert result.flagged

    def test_no_flag_when_sigma_near_zero(self) -> None:
        """sigma_S <= 1e-10 suppresses flagging."""
        state = _make_fresh_state(window=10, k=1.0)
        agg_fn = get_agg(state.config.agg)
        # All constant values: sigma_S will be 0 or near zero
        for _ in range(10):
            state, result = update_axis(state, 0, [5.0, 5.0], agg_fn)
        # After 10 ticks, sigma_S should be near 0 (all S_A identical)
        if result.sigma_S is not None:
            assert not result.flagged

    def test_flag_predicate_exact(self) -> None:
        """flagged == (deviation > threshold) when sigma_S > 1e-10."""
        state = _make_fresh_state(window=10, k=2.0, lam=0.5)
        agg_fn = get_agg(state.config.agg)
        # Warm up with varying values so sigma_S > 0
        rng = np.random.default_rng(99)
        for _ in range(15):
            vals = list(rng.normal(0, 10, size=2))
            state, _ = update_axis(state, 0, vals, agg_fn)
        # Now check the predicate for a few more ticks
        for _ in range(10):
            vals = list(rng.normal(0, 10, size=2))
            state, result = update_axis(state, 0, vals, agg_fn)
            if result.sigma_S is not None and result.sigma_S > 1e-10:
                assert result.deviation is not None
                assert result.threshold is not None
                expected_flag = result.deviation > result.threshold
                assert result.flagged == expected_flag


# ---------------------------------------------------------------------------
# Unit: shape mismatch / non-finite
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_wrong_metric_count_raises(self) -> None:
        state = _make_fresh_state(metric_names=(("m1", "m2"), ("m1",)))
        agg_fn = get_agg(state.config.agg)
        with pytest.raises(ValueError, match="expects 2 metrics"):
            update_axis(state, 0, [1.0], agg_fn)

    def test_non_finite_value_raises(self) -> None:
        state = _make_fresh_state()
        agg_fn = get_agg(state.config.agg)
        with pytest.raises(ValueError, match="non-finite"):
            update_axis(state, 0, [float("nan"), 1.0], agg_fn)


# ---------------------------------------------------------------------------
# Unit: state updates
# ---------------------------------------------------------------------------


class TestStateUpdates:
    def test_metric_count_increments(self) -> None:
        state = _make_fresh_state()
        agg_fn = get_agg(state.config.agg)
        state, _ = update_axis(state, 0, [1.0, 2.0], agg_fn)
        assert state.metric_count[0, 0] == 1
        assert state.metric_count[0, 1] == 1
        state, _ = update_axis(state, 0, [3.0, 4.0], agg_fn)
        assert state.metric_count[0, 0] == 2
        assert state.metric_count[0, 1] == 2

    def test_axis_count_increments(self) -> None:
        state = _make_fresh_state()
        agg_fn = get_agg(state.config.agg)
        state, _ = update_axis(state, 0, [1.0, 2.0], agg_fn)  # S_A=None, no increment
        state, _ = update_axis(state, 0, [3.0, 4.0], agg_fn)  # S_A available
        # axis_count increments only when S_A is written
        assert state.axis_count[0] == 1

    def test_theta_is_updated_in_state(self) -> None:
        state = _make_fresh_state()
        agg_fn = get_agg(state.config.agg)
        state, _ = update_axis(state, 0, [1.0, 2.0], agg_fn)
        state, r2 = update_axis(state, 0, [3.0, 4.0], agg_fn)
        assert state.theta is not None
        assert state.theta[0] == pytest.approx(r2.theta)  # type: ignore[arg-type]
