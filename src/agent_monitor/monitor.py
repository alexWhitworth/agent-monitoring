"""Monitor facade (F-011, API-017) — the only stateful class.

Delegates all detection logic to the pure core. Records tick_flagged
on sessions absorbed in the same tick. No detection logic of its own.
"""

from __future__ import annotations

__all__ = ["Monitor"]

from pathlib import Path

from agent_monitor.advance import advance
from agent_monitor.checkpoint import load_checkpoint, save_checkpoint
from agent_monitor.population import absorb as _absorb
from agent_monitor.population import score_sessions
from agent_monitor.queue import build_review_queue
from agent_monitor.types import (
    MonitorConfig,
    MonitorState,
    QueueConfig,
    ReviewQueuePlan,
    SessionFeatures,
    SessionScore,
    TickMetrics,
    TickResult,
)


class Monitor:
    """Stateful facade over the pure AMDM core.

    Holds the current MonitorState and delegates all operations to pure
    functions. The only stateful concern is recording tick_flagged from
    the most recent update() for absorb() calls in the same tick.
    """

    def __init__(self, config: MonitorConfig) -> None:
        """Initialize a fresh monitor from a validated config.

        Args:
            config: Validated MonitorConfig (from validate_monitor_config).
        """
        A = len(config.axis_names)
        M_max = max(len(m) for m in config.metric_names)
        w = config.window

        import numpy as np

        from agent_monitor.types import SessionWindow

        pop = SessionWindow(
            ids=np.array([], dtype=object),
            ticks=np.array([], dtype=np.int64),
            features=np.empty((0, A), dtype=np.float64),
            tick_flagged=np.array([], dtype=bool),
            absorbed_at=np.array([], dtype=np.int64),
        )

        self._state = MonitorState(
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
        self._last_alert: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, tick: TickMetrics) -> TickResult:
        """Advance one tick and record alert status for subsequent absorb().

        Args:
            tick: Per-tick metric matrix.

        Returns:
            TickResult for this tick.
        """
        self._state, result = advance(self._state, tick)
        self._last_alert = result.alert
        return result

    def absorb(self, sessions: SessionFeatures) -> None:
        """Absorb session rows for Tier 2, recording tick_flagged from the
        most recent update() call.

        Args:
            sessions: Candidate sessions for the current tick.
        """
        self._state = _absorb(self._state, sessions, tick_flagged=self._last_alert)

    def score(self, sessions: SessionFeatures) -> tuple[SessionScore, ...]:
        """Score candidate sessions against the winsorized population.

        Args:
            sessions: Candidate sessions to score.

        Returns:
            Tuple of SessionScore (d2/p_value None during warm-up).
        """
        return score_sessions(self._state, sessions)

    def review_queue(
        self,
        tick_result: TickResult,
        sessions: tuple[SessionScore, ...],
        qcfg: QueueConfig,
    ) -> ReviewQueuePlan:
        """Build a stratified review queue for the given candidates.

        Args:
            tick_result: Tier 1 result for the tick.
            sessions: Scored candidate sessions.
            qcfg: Review-queue budgets and RNG seed.

        Returns:
            ReviewQueuePlan with disjoint strata.
        """
        return build_review_queue(
            tick_result=tick_result,
            session_scores=sessions,
            state=self._state,
            qcfg=qcfg,
        )

    def save(self, path: Path) -> None:
        """Persist current state to a Parquet checkpoint directory.

        Args:
            path: Target checkpoint directory.
        """
        save_checkpoint(self._state, path)

    @classmethod
    def restore(cls, path: Path) -> Monitor:
        """Restore a monitor from a checkpoint directory.

        Args:
            path: Checkpoint directory with state.parquet + population.parquet.

        Returns:
            New Monitor instance with restored state.
        """
        state = load_checkpoint(path)
        monitor = object.__new__(cls)
        monitor._state = state
        monitor._last_alert = False
        return monitor

    # ------------------------------------------------------------------
    # Accessors (read-only views for testing / inspection)
    # ------------------------------------------------------------------

    @property
    def state(self) -> MonitorState:
        """Current MonitorState (read-only)."""
        return self._state

    @property
    def tick(self) -> int:
        """Current global tick counter."""
        return self._state.tick