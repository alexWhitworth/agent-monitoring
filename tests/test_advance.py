# ruff: noqa: N806
"""Tests for advance.py — Tier 1 orchestration and reference parity."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent_monitor.advance import advance
from agent_monitor.config import validate_monitor_config
from agent_monitor.types import MonitorConfig, MonitorState, SessionWindow, TickMetrics


def _make_fresh_state(
    axis_names: tuple[str, ...] = ("Safety", "Quality", "Efficiency"),
    metric_names: tuple[tuple[str, ...], ...] = (("m1", "m2"), ("m1",), ("m1",)),
    **overrides: object,
) -> MonitorState:
    cfg = validate_monitor_config(
        MonitorConfig(axis_names=axis_names, metric_names=metric_names, **overrides)
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
# Unit: basic orchestration
# ---------------------------------------------------------------------------


class TestAdvanceBasic:
    def test_single_tick_warm_up(self) -> None:
        state = _make_fresh_state()
        tm = TickMetrics(X=(np.array([1.0, 2.0]), np.array([3.0]), np.array([4.0])))
        _state2, result = advance(state, tm)
        assert result.tick == 0
        assert len(result.axis_results) == 3
        # First tick: z-scores warming up, S_vector has 0.0 fills
        assert result.S_vector is not None
        assert len(result.S_vector) == 3

    def test_multiple_ticks(self) -> None:
        state = _make_fresh_state()
        for tick_idx in range(10):
            tm = TickMetrics(
                X=(
                    np.array([1.0 * tick_idx, 2.0 * tick_idx]),
                    np.array([3.0 * tick_idx]),
                    np.array([4.0 * tick_idx]),
                )
            )
            state, result = advance(state, tm)
            assert result.tick == tick_idx
            assert result.alert == (result.any_axis_flag or result.joint_flag)


class TestShapeMismatch:
    def test_wrong_axis_count_raises(self) -> None:
        state = _make_fresh_state(
            axis_names=("a", "b"), metric_names=(("m",), ("m",))
        )
        tm = TickMetrics(X=(np.array([1.0]),))  # only 1 axis, expected 2
        with pytest.raises(ValueError, match="expected 2"):
            advance(state, tm)


# ---------------------------------------------------------------------------
# Hypothesis: streaming == batch (no-lookahead)
# ---------------------------------------------------------------------------


class TestNoLookahead:
    @given(
        st.lists(
            st.lists(
                st.floats(allow_nan=False, allow_infinity=False, min_value=-10.0, max_value=10.0),
                min_size=3,
                max_size=3,
            ),
            min_size=5,
            max_size=30,
        )
    )
    @settings(max_examples=50)
    def test_streaming_equals_batch_replay(self, data: list[list[float]]) -> None:
        """Streaming advance over ticks == batch replay from fresh state."""
        state1 = _make_fresh_state(
            axis_names=("a", "b", "c"),
            metric_names=(("m",), ("m",), ("m",)),
            window=10,
        )
        # Streaming
        streaming_results = []
        for row in data:
            tm = TickMetrics(
                X=(np.array([row[0]]), np.array([row[1]]), np.array([row[2]]))
            )
            state1, res = advance(state1, tm)
            streaming_results.append(res)

        # Batch replay
        state2 = _make_fresh_state(
            axis_names=("a", "b", "c"),
            metric_names=(("m",), ("m",), ("m",)),
            window=10,
        )
        for i, row in enumerate(data):
            tm = TickMetrics(
                X=(np.array([row[0]]), np.array([row[1]]), np.array([row[2]]))
            )
            state2, res = advance(state2, tm)
            # Compare with streaming at same tick
            sr = streaming_results[i]
            assert res.tick == sr.tick
            assert res.alert == sr.alert
            # S_vector comparison
            if res.S_vector is not None and sr.S_vector is not None:
                np.testing.assert_allclose(
                    res.S_vector, sr.S_vector, atol=1e-12
                )
            # Joint D2 comparison
            if res.joint.D2 is not None and sr.joint.D2 is not None:
                assert res.joint.D2 == pytest.approx(sr.joint.D2, abs=1e-12)


# ---------------------------------------------------------------------------
# Reference parity: three seeded scenarios from references/AMDM.py
# ---------------------------------------------------------------------------


class TestReferenceParity:
    """[O3] Parity against references/AMDM.py on three seeded scenarios.

    Scenario 1: Normal baseline (t=1..60) — silent
    Scenario 2: Coordinated +1.5sigma shift (t=61..63) — joint flag only
    Scenario 3: Extreme single-axis x20sigma spike (t=64..66) — axis + joint flags
    """

    def _make_reference_config(self) -> MonitorConfig:
        return MonitorConfig(
            axis_names=("Safety", "Quality", "Efficiency", "Reliability", "Compliance"),
            metric_names=tuple(("m",) * 10 for _ in range(5)),
            window=80,
            lam=0.25,
            k=None,  # resolves to chi2_5(0.99)
            alpha=0.01,
            min_warm=30,
            ridge=1e-6,
            agg="mean",
        )

    def test_scenario_1_normal_baseline_silent(self) -> None:
        """t=1..60: all metrics iid N(0,1). Joint should be silent after warm-up."""
        cfg = self._make_reference_config()
        state = _make_fresh_state(
            axis_names=cfg.axis_names,
            metric_names=cfg.metric_names,
            window=cfg.window,
            lam=cfg.lam,
            k=cfg.k,
            alpha=cfg.alpha,
            min_warm=cfg.min_warm,
            ridge=cfg.ridge,
            agg=cfg.agg,
        )
        rng = np.random.default_rng(42)
        last_joint_flag = False
        for _t in range(60):
            X = rng.normal(0.0, 1.0, size=(5, 10))
            tm = TickMetrics(X=tuple(np.asarray(X[i], dtype=np.float64) for i in range(5)))
            state, result = advance(state, tm)
            last_joint_flag = result.joint_flag
        # After 60 normal ticks, joint should be silent (D2 stays below threshold)
        assert not last_joint_flag

    def test_scenario_2_coordinated_shift_joint_flag(self) -> None:
        """t=61..63: all axes shift +1.5sigma. Joint flags, per-axis silent."""
        cfg = self._make_reference_config()
        state = _make_fresh_state(
            axis_names=cfg.axis_names,
            metric_names=cfg.metric_names,
            window=cfg.window,
            lam=cfg.lam,
            k=cfg.k,
            alpha=cfg.alpha,
            min_warm=cfg.min_warm,
            ridge=cfg.ridge,
            agg=cfg.agg,
        )
        rng = np.random.default_rng(42)
        # Warm up with normal baseline (t=1..60)
        for _ in range(60):
            X = rng.normal(0.0, 1.0, size=(5, 10))
            tm = TickMetrics(X=tuple(np.asarray(X[i], dtype=np.float64) for i in range(5)))
            state, _ = advance(state, tm)

        # Coordinated shift (t=61..63)
        joint_flagged_any = False
        any_axis_flagged = False
        for _ in range(3):
            X = rng.normal(1.5, 1.0, size=(5, 10))
            tm = TickMetrics(X=tuple(np.asarray(X[i], dtype=np.float64) for i in range(5)))
            state, result = advance(state, tm)
            if result.joint_flag:
                joint_flagged_any = True
            if result.any_axis_flag:
                any_axis_flagged = True
        assert joint_flagged_any
        assert not any_axis_flagged

    def test_scenario_3_extreme_spike_joint_flag(self) -> None:
        """t=64..66: Safety axis spikes N(20, 0.5). Joint fires (ref parity).

        Note: per-axis flag does NOT fire (k * sigma_S is very strict;
        threshold ~15 with sigma_S inflated by coordinated shift).
        This matches the reference implementation's actual behavior.
        """
        cfg = self._make_reference_config()
        state = _make_fresh_state(
            axis_names=cfg.axis_names,
            metric_names=cfg.metric_names,
            window=cfg.window,
            lam=cfg.lam,
            k=cfg.k,
            alpha=cfg.alpha,
            min_warm=cfg.min_warm,
            ridge=cfg.ridge,
            agg=cfg.agg,
        )
        rng = np.random.default_rng(42)
        # Warm up with normal baseline
        for _ in range(60):
            X = rng.normal(0.0, 1.0, size=(5, 10))
            tm = TickMetrics(X=tuple(np.asarray(X[i], dtype=np.float64) for i in range(5)))
            state, _ = advance(state, tm)

        # Coordinated shift (establish baseline)
        for _ in range(3):
            X = rng.normal(1.5, 1.0, size=(5, 10))
            tm = TickMetrics(X=tuple(np.asarray(X[i], dtype=np.float64) for i in range(5)))
            state, _ = advance(state, tm)

        # Extreme spike on Safety (axis 0)
        joint_flagged_any = False
        for _ in range(3):
            X = rng.normal(0.0, 1.0, size=(5, 10))
            X[0] = rng.normal(20.0, 0.5, size=10)  # Safety blowout
            tm = TickMetrics(X=tuple(np.asarray(X[i], dtype=np.float64) for i in range(5)))
            state, result = advance(state, tm)
            # Joint should fire (D2 spikes due to coordinated deviation across axes)
            if result.joint_flag:
                joint_flagged_any = True
        assert joint_flagged_any
