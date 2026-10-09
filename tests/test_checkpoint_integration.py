# ruff: noqa: N806
"""Integration tests for checkpoint persistence (F-010).

[D2] save -> load round-trip: arrays byte-equal, writeable=False,
config dataclass-equal, and the next advance produces identical results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agent_monitor.advance import advance
from agent_monitor.checkpoint import load_checkpoint, save_checkpoint
from agent_monitor.config import validate_monitor_config
from agent_monitor.population import absorb
from agent_monitor.types import (
    MonitorConfig,
    MonitorState,
    SessionFeatures,
    SessionWindow,
    TickMetrics,
)


def _make_state(
    config: MonitorConfig,
    tick: int = 80,
) -> MonitorState:
    """Build a warm state with filled buffers and non-trivial joint state."""
    A = len(config.axis_names)
    M_max = max(len(m) for m in config.metric_names)
    w = config.window

    # Pre-fill ring buffers with synthetic data
    rng = np.random.default_rng(42)
    metric_buf = rng.normal(0, 1, size=(A, M_max, w)).astype(np.float64)
    metric_count = np.full((A, M_max), w, dtype=np.int64)
    theta = rng.normal(0, 0.5, size=(A,)).astype(np.float64)
    axis_buf = rng.normal(0, 1, size=(A, w)).astype(np.float64)
    axis_count = np.full((A,), w, dtype=np.int64)
    n_joint = 100
    mu = rng.normal(0, 0.5, size=(A,)).astype(np.float64)
    M2 = np.eye(A, dtype=np.float64) * 0.5

    pop = SessionWindow(
        ids=np.array([], dtype=object),
        ticks=np.array([], dtype=np.int64),
        features=np.empty((0, A), dtype=np.float64),
        tick_flagged=np.array([], dtype=bool),
        absorbed_at=np.array([], dtype=np.int64),
    )

    return MonitorState(
        config=config,
        tick=tick,
        metric_buf=metric_buf,
        metric_count=metric_count,
        theta=theta,
        axis_buf=axis_buf,
        axis_count=axis_count,
        n=n_joint,
        mu=mu,
        M2=M2,
        population=pop,
    )


class TestD2RoundTrip:
    """[D2] save -> load -> advance yields identical results."""

    def test_round_trip_preserves_state_identity(self, tmp_path: Path) -> None:
        config = validate_monitor_config(
            MonitorConfig(
                axis_names=("Safety", "Quality"),
                metric_names=(("m1", "m2"), ("m1",)),
                window=80,
                lam=0.25,
                min_warm=30,
                population_window=80,
                min_population=5,
            )
        )
        state = _make_state(config, tick=80)
        cp = tmp_path / "checkpoint"
        save_checkpoint(state, cp)
        restored = load_checkpoint(cp)

        # [D2] config dataclass-equal
        assert restored.config == state.config

        # [D2] arrays byte-equal
        np.testing.assert_array_equal(restored.metric_buf, state.metric_buf)
        np.testing.assert_array_equal(restored.metric_count, state.metric_count)
        assert restored.theta is not None and state.theta is not None
        np.testing.assert_array_equal(restored.theta, state.theta)
        np.testing.assert_array_equal(restored.axis_buf, state.axis_buf)
        np.testing.assert_array_equal(restored.axis_count, state.axis_count)
        assert restored.n == state.n
        np.testing.assert_array_equal(restored.mu, state.mu)
        np.testing.assert_array_equal(restored.M2, state.M2)
        assert restored.tick == state.tick

        # [D2] arrays are read-only after load
        assert not restored.metric_buf.flags.writeable
        assert not restored.mu.flags.writeable
        assert not restored.M2.flags.writeable
        assert not restored.population.features.flags.writeable

    def test_advance_after_round_trip_identical(self, tmp_path: Path) -> None:
        """[D2] The next advance after round-trip is identical within 1e-12."""
        config = validate_monitor_config(
            MonitorConfig(
                axis_names=("Safety", "Quality"),
                metric_names=(("m1",), ("m1",)),
                window=80,
                lam=0.25,
                min_warm=30,
                population_window=80,
                min_population=5,
            )
        )
        state = _make_state(config, tick=80)
        cp = tmp_path / "checkpoint"
        save_checkpoint(state, cp)
        restored = load_checkpoint(cp)

        # Build a tick input (1 metric per axis to match metric_names)
        X_a = np.array([1.5], dtype=np.float64)
        X_b = np.array([-0.5], dtype=np.float64)
        tick_input = TickMetrics(X=(X_a, X_b))

        # Advance from original state
        result1 = advance(state, tick_input)
        # Advance from restored state
        result2 = advance(restored, tick_input)

        # State fields should be identical
        assert result1[0].tick == result2[0].tick
        assert result1[0].n == result2[0].n
        np.testing.assert_allclose(result1[0].mu, result2[0].mu, atol=1e-12)
        np.testing.assert_allclose(result1[0].M2, result2[0].M2, atol=1e-12)
        np.testing.assert_allclose(result1[0].metric_buf, result2[0].metric_buf, atol=1e-12)

        # Results should be identical
        t1, t2 = result1[1], result2[1]
        assert t1.tick == t2.tick
        assert t1.alert == t2.alert
        assert t1.any_axis_flag == t2.any_axis_flag
        assert t1.joint_flag == t2.joint_flag
        # Axis results
        for a1, a2 in zip(t1.axis_results, t2.axis_results):
            assert a1.axis == a2.axis
            assert a1.flagged == a2.flagged
            if a1.S_A is not None and a2.S_A is not None:
                assert a1.S_A == pytest.approx(a2.S_A, abs=1e-12)

    def test_round_trip_with_population_and_advance(self, tmp_path: Path) -> None:
        """Round-trip a state that has a populated SessionWindow, then advance."""
        config = validate_monitor_config(
            MonitorConfig(
                axis_names=("Safety", "Quality"),
                metric_names=(("m1",), ("m1",)),
                window=80,
                lam=0.25,
                min_warm=30,
                population_window=100,
                min_population=5,
            )
        )
        A = 2
        state = _make_state(config, tick=80)

        # Absorb population rows
        rng = np.random.default_rng(99)
        pop_n = 50
        features = rng.normal(0, 1, size=(pop_n, A)).astype(np.float64)
        sf = SessionFeatures(
            ids=tuple(f"pop_{i}" for i in range(pop_n)),
            tick=75,
            F=features,
        )
        state = absorb(state, sf, tick_flagged=False)

        cp = tmp_path / "checkpoint"
        save_checkpoint(state, cp)
        restored = load_checkpoint(cp)

        # Verify population
        np.testing.assert_array_equal(restored.population.ids, state.population.ids)
        np.testing.assert_array_equal(restored.population.features, state.population.features)
        np.testing.assert_array_equal(restored.population.tick_flagged, state.population.tick_flagged)
        assert not restored.population.features.flags.writeable

        # Advance both
        X_a = np.array([0.5], dtype=np.float64)
        X_b = np.array([-1.0], dtype=np.float64)
        tick_input = TickMetrics(X=(X_a, X_b))

        state2, res1 = advance(state, tick_input)
        restored2, res2 = advance(restored, tick_input)

        assert res1.alert == res2.alert
        np.testing.assert_allclose(restored2.mu, state2.mu, atol=1e-12)
        np.testing.assert_allclose(restored2.M2, state2.M2, atol=1e-12)

    def test_multiple_advances_after_round_trip(self, tmp_path: Path) -> None:
        """Multiple advances after restore remain equivalent."""
        config = validate_monitor_config(
            MonitorConfig(
                axis_names=("a", "b"),
                metric_names=(("m",), ("m",)),
                window=80,
                lam=0.25,
                min_warm=10,
                population_window=80,
                min_population=5,
            )
        )
        state = _make_state(config, tick=80)
        cp = tmp_path / "checkpoint"
        save_checkpoint(state, cp)
        restored = load_checkpoint(cp)

        rng = np.random.default_rng(1)
        for i in range(20):
            X_a = rng.normal(0, 1, size=(1,)).astype(np.float64)
            X_b = rng.normal(0, 1, size=(1,)).astype(np.float64)
            tick_input = TickMetrics(X=(X_a, X_b))

            state, res1 = advance(state, tick_input)
            restored, res2 = advance(restored, tick_input)

            assert res1.alert == res2.alert, f"Tick {i}: alerts differ"
            assert res1.joint_flag == res2.joint_flag, f"Tick {i}: joint_flag differs"
            np.testing.assert_allclose(state.mu, restored.mu, atol=1e-12)
            np.testing.assert_allclose(state.M2, restored.M2, atol=1e-12)