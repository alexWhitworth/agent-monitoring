"""Unit tests for Monitor facade (F-011) — delegation correctness, tick_flagged
recording, restore/save passthrough, and edge cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agent_monitor.advance import advance
from agent_monitor.checkpoint import load_checkpoint, save_checkpoint
from agent_monitor.config import validate_monitor_config
from agent_monitor.monitor import Monitor
from agent_monitor.population import absorb as pure_absorb
from agent_monitor.types import (
    MonitorConfig,
    MonitorState,
    QueueConfig,
    SessionFeatures,
    SessionScore,
    TickMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    axis_names: tuple[str, ...] = ("Safety", "Quality"),
    metric_names: tuple[tuple[str, ...], ...] = (("m1",), ("m1",)),
    window: int = 80,
    lam: float = 0.25,
    min_warm: int = 2,
    **overrides: object,
) -> MonitorConfig:
    return validate_monitor_config(
        MonitorConfig(
            axis_names=axis_names,
            metric_names=metric_names,
            window=window,
            lam=lam,
            min_warm=min_warm,
            **overrides,
        )
    )


def _make_tick(a_val: float = 1.0, b_val: float = 0.0) -> TickMetrics:
    return TickMetrics(
        X=(np.array([a_val], dtype=np.float64), np.array([b_val], dtype=np.float64))
    )


# ---------------------------------------------------------------------------
# Delegation correctness
# ---------------------------------------------------------------------------


class TestDelegation:
    def test_update_matches_advance(self) -> None:
        config = _make_config()
        monitor = Monitor(config)

        # Build equivalent pure state
        A = len(config.axis_names)
        M_max = max(len(m) for m in config.metric_names)
        w = config.window
        from agent_monitor.types import SessionWindow
        pop = SessionWindow(
            ids=np.array([], dtype=object),
            ticks=np.array([], dtype=np.int64),
            features=np.empty((0, A), dtype=np.float64),
            tick_flagged=np.array([], dtype=bool),
            absorbed_at=np.array([], dtype=np.int64),
        )
        pure_state = MonitorState(
            config=config, tick=0,
            metric_buf=np.zeros((A, M_max, w), dtype=np.float64),
            metric_count=np.zeros((A, M_max), dtype=np.int64),
            theta=None,
            axis_buf=np.zeros((A, w), dtype=np.float64),
            axis_count=np.zeros(A, dtype=np.int64),
            n=0, mu=np.zeros(A, dtype=np.float64),
            M2=np.zeros((A, A), dtype=np.float64),
            population=pop,
        )

        rng = np.random.default_rng(42)
        for i in range(100):
            tick = TickMetrics(
                X=(
                    rng.normal(0, 0.5, size=(1,)).astype(np.float64),
                    rng.normal(0, 0.5, size=(1,)).astype(np.float64),
                )
            )
            m_result = monitor.update(tick)
            pure_state, p_result = advance(pure_state, tick)

            assert m_result.tick == p_result.tick
            assert m_result.alert == p_result.alert
            assert m_result.joint_flag == p_result.joint_flag
            assert m_result.any_axis_flag == p_result.any_axis_flag
            np.testing.assert_allclose(monitor.state.mu, pure_state.mu, atol=1e-12)
            np.testing.assert_allclose(monitor.state.M2, pure_state.M2, atol=1e-12)

    def test_absorb_delegates_to_pure(self) -> None:
        config = _make_config()
        monitor = Monitor(config)

        # First update so absorb has tick_flagged context
        tick = _make_tick(10.0, 0.0)  # big value to trigger alert
        monitor.update(tick)

        sessions = SessionFeatures(
            ids=("a", "b"), tick=monitor.tick - 1, F=np.array([[1.0, 2.0], [3.0, 4.0]])
        )
        monitor.absorb(sessions)
        assert len(monitor.state.population.ids) == 2
        # tick_flagged should match last alert
        assert monitor.state.population.tick_flagged[0] == monitor._last_alert

    def test_score_delegates(self) -> None:
        config = _make_config(min_population=2)
        monitor = Monitor(config)

        # Absorb some population rows
        sessions = SessionFeatures(
            ids=("a", "b", "c", "d"),
            tick=0,
            F=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        )
        monitor.absorb(sessions)

        scores = monitor.score(
            SessionFeatures(ids=("x",), tick=0, F=np.array([[0.0, 0.0]]))
        )
        assert len(scores) == 1
        assert scores[0].d2 is not None

    def test_review_queue_delegates(self) -> None:
        config = _make_config()
        monitor = Monitor(config)
        tick = _make_tick()
        tr = monitor.update(tick)

        sessions = SessionFeatures(
            ids=("a", "b"), tick=0, F=np.array([[1.0, 2.0], [3.0, 4.0]])
        )
        monitor.absorb(sessions)
        scores = (
            SessionScore(id="a", d2=10.0, p_value=0.01),
            SessionScore(id="b", d2=5.0, p_value=0.1),
        )
        qcfg = QueueConfig(top_k=1, flagged=0, per_axis=0, uniform=1, seed=0)
        plan = monitor.review_queue(tr, scores, qcfg)
        assert plan.strata["top_k"] == ("a",)
        assert "b" in plan.strata["uniform"]


# ---------------------------------------------------------------------------
# Tick-flagged recording
# ---------------------------------------------------------------------------


class TestTickFlagged:
    def test_absorb_records_last_alert(self) -> None:
        config = _make_config()
        monitor = Monitor(config)

        # Non-alerting tick
        tick_normal = _make_tick(0.0, 0.0)
        result_normal = monitor.update(tick_normal)
        sessions = SessionFeatures(
            ids=("normal_sess",), tick=monitor.tick - 1,
            F=np.array([[1.0, 2.0]]),
        )
        monitor.absorb(sessions)
        assert monitor.state.population.tick_flagged[0] == result_normal.alert

    def test_absorb_records_alert_tick_flagged(self) -> None:
        config = _make_config()
        monitor = Monitor(config)

        # Drive the state to a warm condition, then inject a spike
        for _ in range(10):
            monitor.update(_make_tick(0.0, 0.0))

        # Spike tick
        spike = TickMetrics(
            X=(np.array([100.0], dtype=np.float64), np.array([0.0], dtype=np.float64))
        )
        result_spike = monitor.update(spike)
        sessions = SessionFeatures(
            ids=("spike_sess",), tick=monitor.tick - 1,
            F=np.array([[100.0, 0.0]]),
        )
        monitor.absorb(sessions)
        # The spike's alert status should be recorded
        assert monitor.state.population.tick_flagged[-1] == result_spike.alert

    def test_absorb_before_update_defaults_false(self) -> None:
        """Absorb before any update() call should default tick_flagged=False."""
        config = _make_config()
        monitor = Monitor(config)
        sessions = SessionFeatures(
            ids=("early",), tick=0, F=np.array([[1.0, 2.0]]),
        )
        monitor.absorb(sessions)
        assert bool(monitor.state.population.tick_flagged[0]) is False

    def test_tick_flagged_persists_across_multiple_absorbs(self) -> None:
        """tick_flagged is set from the most recent update(), used for all absorbs until next update."""
        config = _make_config()
        monitor = Monitor(config)

        # Alerting tick
        spike = TickMetrics(
            X=(np.array([100.0], dtype=np.float64), np.array([0.0], dtype=np.float64))
        )
        for _ in range(5):
            monitor.update(_make_tick(0.0, 0.0))  # warm up
        monitor.update(spike)  # this should be alerting

        # Two absorbs
        monitor.absorb(SessionFeatures(ids=("a",), tick=monitor.tick - 1, F=np.array([[5.0, 0.0]])))
        monitor.absorb(SessionFeatures(ids=("b",), tick=monitor.tick - 1, F=np.array([[10.0, 0.0]])))
        # Both should have the same tick_flagged from the spike
        assert monitor.state.population.tick_flagged[-2] == monitor.state.population.tick_flagged[-1]


# ---------------------------------------------------------------------------
# Restore / save round-trip
# ---------------------------------------------------------------------------


class TestRestoreSave:
    def test_save_and_restore_round_trip(self, tmp_path: Path) -> None:
        config = _make_config()
        monitor = Monitor(config)
        for _ in range(5):
            monitor.update(_make_tick(0.5, -0.5))

        cp = tmp_path / "checkpoint"
        monitor.save(cp)

        restored = Monitor.restore(cp)
        assert restored.state.config == monitor.state.config
        assert restored.state.tick == monitor.state.tick
        np.testing.assert_array_equal(restored.state.mu, monitor.state.mu)
        np.testing.assert_array_equal(restored.state.M2, monitor.state.M2)

    def test_restored_monitor_continues_correctly(self, tmp_path: Path) -> None:
        config = _make_config()
        monitor = Monitor(config)
        for _ in range(10):
            monitor.update(_make_tick(0.0, 0.0))

        cp = tmp_path / "checkpoint"
        monitor.save(cp)
        restored = Monitor.restore(cp)

        # Advance both
        tick = _make_tick(2.0, -1.0)
        r1 = monitor.update(tick)
        r2 = restored.update(tick)
        assert r1.alert == r2.alert
        assert r1.joint_flag == r2.joint_flag
        np.testing.assert_allclose(monitor.state.mu, restored.state.mu, atol=1e-12)

    def test_restore_resets_last_alert_to_false(self, tmp_path: Path) -> None:
        config = _make_config()
        monitor = Monitor(config)
        monitor.update(_make_tick(100.0, 0.0))  # alerting
        cp = tmp_path / "checkpoint"
        monitor.save(cp)

        restored = Monitor.restore(cp)
        assert restored._last_alert is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_absorb_before_first_update(self) -> None:
        config = _make_config()
        monitor = Monitor(config)
        sessions = SessionFeatures(
            ids=("a",), tick=0, F=np.array([[1.0, 2.0]]),
        )
        monitor.absorb(sessions)
        assert bool(monitor.state.population.tick_flagged[0]) is False

    def test_save_without_updates(self, tmp_path: Path) -> None:
        config = _make_config()
        monitor = Monitor(config)
        cp = tmp_path / "checkpoint"
        monitor.save(cp)
        restored = Monitor.restore(cp)
        assert restored.tick == 0

    def test_score_without_population(self) -> None:
        config = _make_config(min_population=100)
        monitor = Monitor(config)
        scores = monitor.score(
            SessionFeatures(ids=("x",), tick=0, F=np.array([[1.0, 2.0]]))
        )
        assert scores[0].d2 is None

    def test_review_queue_empty_candidates(self) -> None:
        config = _make_config()
        monitor = Monitor(config)
        tr = monitor.update(_make_tick())
        plan = monitor.review_queue(tr, (), QueueConfig())
        assert plan.strata["top_k"] == ()
        assert plan.unallocated == ()

    def test_multiple_update_absorb_cycles(self) -> None:
        """Simulate a realistic tick-by-tick cycle."""
        config = _make_config(min_population=3)
        monitor = Monitor(config)

        for i in range(20):
            tick = TickMetrics(
                X=(
                    np.array([float(i % 5)], dtype=np.float64),
                    np.array([float((i + 1) % 5)], dtype=np.float64),
                )
            )
            tr = monitor.update(tick)
            sessions = SessionFeatures(
                ids=(f"sess_{i}",),
                tick=monitor.tick - 1,
                F=np.array([[float(i), float(i + 1)]], dtype=np.float64),
            )
            monitor.absorb(sessions)

        assert monitor.tick == 20
        assert len(monitor.state.population.ids) == 20