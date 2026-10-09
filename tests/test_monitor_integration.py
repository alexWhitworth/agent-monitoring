"""Integration tests for Monitor facade (F-011) — Airflow-style
restore -> advance -> absorb -> score -> save workflow.

Validates that the Monitor's pure-core delegation produces identical
results to direct pure-function calls, and that the full workflow
(including checkpoint round-trips) is correct.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agent_monitor.advance import advance
from agent_monitor.checkpoint import load_checkpoint
from agent_monitor.config import validate_monitor_config
from agent_monitor.monitor import Monitor
from agent_monitor.population import absorb as pure_absorb
from agent_monitor.population import score_sessions
from agent_monitor.types import MonitorConfig, MonitorState, SessionFeatures, SessionWindow, TickMetrics


def _make_config(**overrides: object) -> MonitorConfig:
    return validate_monitor_config(
        MonitorConfig(
            axis_names=("Safety", "Quality"),
            metric_names=(("m1",), ("m1",)),
            window=80,
            lam=0.25,
            min_warm=5,
            population_window=80,
            min_population=5,
            **overrides,
        )
    )


class TestAirflowWorkflow:
    """End-to-end Airflow-style: restore -> advance -> absorb -> score -> save."""

    def test_full_workflow_cycle(self, tmp_path: Path) -> None:
        config = _make_config()
        monitor = Monitor(config)

        # Simulate a day's worth of ticks with sessions
        rng = np.random.default_rng(42)
        for day in range(3):
            # --- Start of day: restore from checkpoint ---
            if day > 0:
                monitor = Monitor.restore(cp)

            tick_metrics = [
                TickMetrics(
                    X=(
                        rng.normal(0, 0.3, size=(1,)).astype(np.float64),
                        rng.normal(0, 0.3, size=(1,)).astype(np.float64),
                    )
                )
                for _ in range(10)
            ]

            for tm in tick_metrics:
                # --- advance ---
                tick_result = monitor.update(tm)

                # --- absorb sessions for this tick ---
                n_sessions = rng.integers(1, 5)
                session_features = rng.normal(0, 1, size=(n_sessions, 2)).astype(np.float64)
                sessions = SessionFeatures(
                    ids=tuple(f"day{day}_tick{monitor.tick - 1}_{j}" for j in range(n_sessions)),
                    tick=monitor.tick - 1,
                    F=session_features,
                )
                monitor.absorb(sessions)

                # --- score the sessions ---
                scores = monitor.score(sessions)

                # --- build review queue ---
                from agent_monitor.types import QueueConfig
                qcfg = QueueConfig(top_k=2, flagged=1, per_axis=2, uniform=1, seed=day)
                plan = monitor.review_queue(tick_result, scores, qcfg)

                # Verify basic queue invariants
                all_sel: set[str] = set()
                for s in ("top_k", "flagged", "per_axis", "uniform"):
                    all_sel.update(plan.strata[s])
                assert plan.strata["reserved"] == ()
                assert all_sel | set(plan.unallocated) == set(s.id for s in scores)

            # --- End of day: save checkpoint ---
            cp = tmp_path / f"checkpoint_day{day}"
            monitor.save(cp)

            # Verify checkpoint is loadable
            restored = Monitor.restore(cp)
            assert restored.tick == monitor.tick
            np.testing.assert_array_equal(restored.state.mu, monitor.state.mu)
            np.testing.assert_array_equal(restored.state.population.ids, monitor.state.population.ids)

    def test_monitor_vs_pure_equivalence_over_workflow(self) -> None:
        """Monitor facade produces identical results to pure function calls."""
        config = _make_config()
        monitor = Monitor(config)

        # Build equivalent pure state
        A = 2
        w = config.window
        pure_state = MonitorState(
            config=config, tick=0,
            metric_buf=np.zeros((A, 1, w), dtype=np.float64),
            metric_count=np.zeros((A, 1), dtype=np.int64),
            theta=None,
            axis_buf=np.zeros((A, w), dtype=np.float64),
            axis_count=np.zeros(A, dtype=np.int64),
            n=0, mu=np.zeros(A, dtype=np.float64),
            M2=np.zeros((A, A), dtype=np.float64),
            population=SessionWindow(
                ids=np.array([], dtype=object),
                ticks=np.array([], dtype=np.int64),
                features=np.empty((0, A), dtype=np.float64),
                tick_flagged=np.array([], dtype=bool),
                absorbed_at=np.array([], dtype=np.int64),
            ),
        )

        rng = np.random.default_rng(99)
        for i in range(30):
            tick = TickMetrics(
                X=(
                    rng.normal(0, 0.5, size=(1,)).astype(np.float64),
                    rng.normal(0, 0.5, size=(1,)).astype(np.float64),
                )
            )

            # Monitor path
            m_result = monitor.update(tick)

            # Pure path
            pure_state, p_result = advance(pure_state, tick)

            assert m_result.alert == p_result.alert
            assert m_result.joint_flag == p_result.joint_flag
            np.testing.assert_allclose(monitor.state.mu, pure_state.mu, atol=1e-12)

            # Absorb sessions via both paths
            n_sess = rng.integers(1, 3)
            f = rng.normal(0, 1, size=(n_sess, 2)).astype(np.float64)
            sessions = SessionFeatures(
                ids=tuple(f"sess_{i}_{j}" for j in range(n_sess)),
                tick=monitor.tick - 1,
                F=f,
            )
            monitor.absorb(sessions)
            pure_state = pure_absorb(pure_state, sessions, tick_flagged=m_result.alert)

            # Score via both paths
            cand = SessionFeatures(
                ids=("cand",), tick=monitor.tick - 1,
                F=rng.normal(0, 1, size=(1, 2)).astype(np.float64),
            )
            m_scores = monitor.score(cand)
            p_scores = score_sessions(pure_state, cand)

            if m_scores[0].d2 is not None and p_scores[0].d2 is not None:
                assert m_scores[0].d2 == pytest.approx(p_scores[0].d2, abs=1e-12)

        # Final state equivalence
        np.testing.assert_array_equal(monitor.state.population.ids, pure_state.population.ids)
        np.testing.assert_allclose(monitor.state.mu, pure_state.mu, atol=1e-12)
        np.testing.assert_allclose(monitor.state.M2, pure_state.M2, atol=1e-12)