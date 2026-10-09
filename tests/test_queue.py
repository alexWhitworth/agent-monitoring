# ruff: noqa: N806
"""Tests for review queue stratification (F-009) — five-stratum allocation,
disjointness, shortfall cascade, determinism, and edge cases.

API-014: build_review_queue(tick_result, session_scores, state, qcfg) -> ReviewQueuePlan
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent_monitor.config import validate_monitor_config
from agent_monitor.population import absorb
from agent_monitor.queue import build_review_queue
from agent_monitor.types import (
    AxisResult,
    JointResult,
    MonitorConfig,
    MonitorState,
    QueueConfig,
    SessionFeatures,
    SessionScore,
    SessionWindow,
    TickResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fresh_state(
    axis_names: tuple[str, ...] = ("Safety", "Quality"),
    metric_names: tuple[tuple[str, ...], ...] = (("m",), ("m",)),
    population_window: int = 200,
    winsor_pct: float | None = None,
    winsor_k: int = 3,
    min_population: int = 5,
    window: int = 80,
    tick: int = 0,
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


def _make_tick_result(
    tick: int = 0,
    axis_results: tuple[AxisResult, ...] | None = None,
    joint_flagged: bool = False,
    any_axis_flag: bool = False,
) -> TickResult:
    if axis_results is None:
        axis_results = (
            AxisResult(
                axis="Safety", S_A=0.0, theta=0.0, sigma_S=1.0,
                deviation=0.0, threshold=3.0, flagged=False, note="ok",
            ),
            AxisResult(
                axis="Quality", S_A=0.0, theta=0.0, sigma_S=1.0,
                deviation=0.0, threshold=3.0, flagged=False, note="ok",
            ),
        )
    alert = any_axis_flag or joint_flagged
    return TickResult(
        tick=tick,
        axis_results=axis_results,
        joint=JointResult(
            n=0, D2=None, threshold=5.99, p_value=None,
            flagged=joint_flagged, note="ok",
        ),
        S_vector=np.array([0.0, 0.0]),
        any_axis_flag=any_axis_flag,
        joint_flag=joint_flagged,
        alert=alert,
    )


def _absorb_sessions(
    state: MonitorState, session_ids: list[str], features: np.ndarray, tick_flagged: bool = False
) -> MonitorState:
    """Absorb sessions into state so they appear in the population."""
    sf = SessionFeatures(ids=tuple(session_ids), tick=state.tick, F=features)
    return absorb(state, sf, tick_flagged=tick_flagged)


# ---------------------------------------------------------------------------
# Unit: basic correctness
# ---------------------------------------------------------------------------


class TestBasicCorrectness:
    def test_empty_candidates_returns_empty_plan(self) -> None:
        state = _make_fresh_state()
        tr = _make_tick_result()
        qcfg = QueueConfig(top_k=10, flagged=5, per_axis=4, uniform=3, seed=42)
        plan = build_review_queue(tr, (), state, qcfg)
        assert plan.strata["top_k"] == ()
        assert plan.strata["flagged"] == ()
        assert plan.strata["per_axis"] == ()
        assert plan.strata["uniform"] == ()
        assert plan.strata["reserved"] == ()
        assert plan.unallocated == ()

    def test_strata_disjoint(self) -> None:
        state = _make_fresh_state(tick=0)
        rng = np.random.default_rng(1)
        n = 50
        features = rng.normal(0, 1, size=(n, 2))
        ids = [f"sess_{i}" for i in range(n)]
        state = _absorb_sessions(state, ids, features, tick_flagged=False)
        scores = tuple(
            SessionScore(id=sid, d2=float(features[i, 0] ** 2 + features[i, 1] ** 2), p_value=0.5)
            for i, sid in enumerate(ids)
        )
        tr = _make_tick_result()
        qcfg = QueueConfig(top_k=10, flagged=5, per_axis=6, uniform=10, seed=0)
        plan = build_review_queue(tr, scores, state, qcfg)

        all_selected: set[str] = set()
        for stratum in ("top_k", "flagged", "per_axis", "uniform"):
            stratum_ids = set(plan.strata[stratum])
            # Disjoint with previously seen
            assert len(stratum_ids & all_selected) == 0, f"{stratum} overlaps"
            all_selected.update(stratum_ids)

        # reserved is always empty
        assert plan.strata["reserved"] == ()

        # selected + unallocated == all candidates (set equality)
        assert all_selected | set(plan.unallocated) == set(ids)

    def test_reserved_always_empty(self) -> None:
        state = _make_fresh_state(tick=0)
        features = np.random.default_rng(1).normal(0, 1, size=(100, 2))
        ids = [f"sess_{i}" for i in range(100)]
        state = _absorb_sessions(state, ids, features)
        scores = tuple(
            SessionScore(id=sid, d2=10.0, p_value=0.01) for sid in ids
        )
        tr = _make_tick_result()
        qcfg = QueueConfig(top_k=50, flagged=50, per_axis=50, uniform=50, reserved=50, seed=0)
        plan = build_review_queue(tr, scores, state, qcfg)
        assert plan.strata["reserved"] == ()

    def test_selected_plus_unallocated_equals_candidates(self) -> None:
        state = _make_fresh_state(tick=0)
        rng = np.random.default_rng(42)
        n = 200
        features = rng.normal(0, 1, size=(n, 2))
        ids = [f"sess_{i}" for i in range(n)]
        state = _absorb_sessions(state, ids, features)
        scores = tuple(
            SessionScore(id=sid, d2=float(features[i, 0] ** 2 + features[i, 1] ** 2), p_value=0.5)
            for i, sid in enumerate(ids)
        )
        tr = _make_tick_result()
        qcfg = QueueConfig(top_k=10, flagged=5, per_axis=4, uniform=3, seed=0)
        plan = build_review_queue(tr, scores, state, qcfg)
        selected = set()
        for s in ("top_k", "flagged", "per_axis", "uniform"):
            selected.update(plan.strata[s])
        assert selected | set(plan.unallocated) == set(ids)
        assert len(selected & set(plan.unallocated)) == 0


# ---------------------------------------------------------------------------
# Unit: budget adherence
# ---------------------------------------------------------------------------


class TestBudgetAdherence:
    def test_stratum_size_respects_budget(self) -> None:
        state = _make_fresh_state(tick=0)
        rng = np.random.default_rng(7)
        n = 100
        features = rng.normal(0, 1, size=(n, 2))
        ids = [f"sess_{i}" for i in range(n)]
        state = _absorb_sessions(state, ids, features)
        scores = tuple(
            SessionScore(id=sid, d2=float(features[i, 0] ** 2 + features[i, 1] ** 2), p_value=0.5)
            for i, sid in enumerate(ids)
        )
        tr = _make_tick_result()
        qcfg = QueueConfig(top_k=7, flagged=3, per_axis=4, uniform=5, seed=0)
        plan = build_review_queue(tr, scores, state, qcfg)
        assert len(plan.strata["top_k"]) <= qcfg.top_k
        # flagged/per_axis/uniform may exceed original budget due to cascade
        cascade_after_tk = max(0, qcfg.top_k - len(plan.strata["top_k"]))
        assert len(plan.strata["flagged"]) <= qcfg.flagged + cascade_after_tk
        cascade_after_fg = max(0, cascade_after_tk + qcfg.flagged - len(plan.strata["flagged"]))
        assert len(plan.strata["per_axis"]) <= qcfg.per_axis + cascade_after_fg
        cascade_after_pa = max(0, cascade_after_fg + qcfg.per_axis - len(plan.strata["per_axis"]))
        assert len(plan.strata["uniform"]) <= qcfg.uniform + cascade_after_pa

    def test_budget_starvation_top_k(self) -> None:
        """When fewer candidates than top_k budget, take all."""
        state = _make_fresh_state(tick=0)
        features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        ids = ["a", "b", "c"]
        state = _absorb_sessions(state, ids, features)
        scores = tuple(
            SessionScore(id=sid, d2=float(i), p_value=0.5) for i, sid in enumerate(ids)
        )
        tr = _make_tick_result()
        qcfg = QueueConfig(top_k=100, flagged=0, per_axis=0, uniform=0, seed=0)
        plan = build_review_queue(tr, scores, state, qcfg)
        assert len(plan.strata["top_k"]) == 3
        assert len(plan.strata["flagged"]) == 0
        assert len(plan.unallocated) == 0

    def test_zero_budgets_all_strata_empty(self) -> None:
        state = _make_fresh_state(tick=0)
        features = np.random.default_rng(1).normal(0, 1, size=(20, 2))
        ids = [f"sess_{i}" for i in range(20)]
        state = _absorb_sessions(state, ids, features)
        scores = tuple(
            SessionScore(id=sid, d2=10.0, p_value=0.01) for sid in ids
        )
        tr = _make_tick_result()
        qcfg = QueueConfig(top_k=0, flagged=0, per_axis=0, uniform=0, seed=0)
        plan = build_review_queue(tr, scores, state, qcfg)
        for s in ("top_k", "flagged", "per_axis", "uniform", "reserved"):
            assert plan.strata[s] == ()
        assert set(plan.unallocated) == set(ids)


# ---------------------------------------------------------------------------
# Unit: shortfall cascade
# ---------------------------------------------------------------------------


class TestShortfallCascade:
    def test_top_k_cascade_to_flagged(self) -> None:
        """Unfilled top_k budget flows to flagged stratum."""
        state = _make_fresh_state(tick=0)
        features = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
        ids = ["a", "b", "c", "d", "e"]
        # Mark sessions a,b as flagged
        state = _absorb_sessions(state, ids[:2], features[:2], tick_flagged=True)
        state = _absorb_sessions(state, ids[2:], features[2:], tick_flagged=False)
        scores = tuple(
            SessionScore(id=sid, d2=float(features[i, 0]), p_value=0.5)
            for i, sid in enumerate(ids)
        )
        tr = _make_tick_result()
        # top_k budget = 2, flagged budget = 2, but cascade should allow more flagged
        qcfg = QueueConfig(top_k=2, flagged=2, per_axis=0, uniform=0, seed=0)
        plan = build_review_queue(tr, scores, state, qcfg)
        assert len(plan.strata["top_k"]) == 2
        # Remaining 3 candidates: only a,b were flagged (but d2=1,2 are lowest),
        # so they won't be in top_k. flagged stratum should get them.
        # With cascade of 0 from top_k (top_k was fully filled), flagged gets 2.
        assert len(plan.strata["flagged"]) == 2
        assert "a" in plan.strata["flagged"] or "b" in plan.strata["flagged"]

    def test_cascade_through_all_strata(self) -> None:
        """Shortfall cascades through all strata."""
        state = _make_fresh_state(tick=0)
        features = np.random.default_rng(99).normal(0, 1, size=(5, 2))
        ids = ["a", "b", "c", "d", "e"]
        # Make all sessions flagged so flagged stratum has candidates
        state = _absorb_sessions(state, ids, features, tick_flagged=True)
        scores = tuple(
            SessionScore(id=sid, d2=float(i), p_value=0.5) for i, sid in enumerate(ids)
        )
        tr = _make_tick_result()
        qcfg = QueueConfig(top_k=1, flagged=1, per_axis=1, uniform=2, seed=0)
        plan = build_review_queue(tr, scores, state, qcfg)
        # top_k=1 fills, flagged=1 fills, per_axis gets 1+0=1 (/2 = 0 each, but with cascade becomes 1)
        # Actually: top_k gets 1 (d2=4), flagged gets 1 (d2=3), cascade=0 from both
        # per_axis=1, floor(1/2)=0 -> 0 selected; cascade=1 to uniform
        # uniform=2+1=3, selects 3 from remaining 3
        assert len(plan.strata["top_k"]) == 1
        assert len(plan.strata["flagged"]) == 1
        # per_axis may get 0 if floor(1/2)=0; that's OK, cascade handles it
        assert len(plan.strata["uniform"]) >= 2
        assert len(plan.unallocated) == 0
        # All 5 sessions allocated
        all_alloc = set()
        for s in ("top_k", "flagged", "per_axis", "uniform"):
            all_alloc.update(plan.strata[s])
        assert len(all_alloc) == 5

    def test_cascade_truncated_by_available_candidates(self) -> None:
        """Cascade can't exceed total remaining candidates."""
        state = _make_fresh_state(tick=0)
        features = np.array([[1.0, 0.0]])
        ids = ["only"]
        state = _absorb_sessions(state, ids, features, tick_flagged=False)
        scores = (SessionScore(id="only", d2=1.0, p_value=0.5),)
        tr = _make_tick_result()
        qcfg = QueueConfig(top_k=0, flagged=0, per_axis=0, uniform=100, seed=0)
        plan = build_review_queue(tr, scores, state, qcfg)
        assert len(plan.strata["uniform"]) == 1
        assert len(plan.unallocated) == 0


# ---------------------------------------------------------------------------
# Unit: determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_same_plan(self) -> None:
        state = _make_fresh_state(tick=0)
        rng = np.random.default_rng(42)
        features = rng.normal(0, 1, size=(50, 2))
        ids = [f"sess_{i}" for i in range(50)]
        state = _absorb_sessions(state, ids, features)
        scores = tuple(
            SessionScore(id=sid, d2=float(features[i, 0] ** 2 + features[i, 1] ** 2), p_value=0.5)
            for i, sid in enumerate(ids)
        )
        tr = _make_tick_result()
        qcfg = QueueConfig(top_k=10, flagged=5, per_axis=4, uniform=10, seed=42)

        plan1 = build_review_queue(tr, scores, state, qcfg)
        plan2 = build_review_queue(tr, scores, state, qcfg)
        assert plan1.strata == plan2.strata
        assert plan1.unallocated == plan2.unallocated

    def test_different_seed_different_uniform(self) -> None:
        """Different seed should produce different uniform stratum given enough candidates."""
        state = _make_fresh_state(tick=0)
        rng = np.random.default_rng(1)
        features = rng.normal(0, 1, size=(100, 2))
        ids = [f"sess_{i}" for i in range(100)]
        state = _absorb_sessions(state, ids, features)
        scores = tuple(
            SessionScore(id=sid, d2=0.0, p_value=0.5) for sid in ids  # all equal d2
        )
        tr = _make_tick_result()

        qcfg_a = QueueConfig(top_k=0, flagged=0, per_axis=0, uniform=20, seed=1)
        qcfg_b = QueueConfig(top_k=0, flagged=0, per_axis=0, uniform=20, seed=2)

        plan_a = build_review_queue(tr, scores, state, qcfg_a)
        plan_b = build_review_queue(tr, scores, state, qcfg_b)
        # Different seeds should produce different uniform selections
        # (Note: with equal d2, top_k is ambiguous but uniform should differ)
        assert plan_a.strata["uniform"] != plan_b.strata["uniform"]


# ---------------------------------------------------------------------------
# Unit: flagged stratum
# ---------------------------------------------------------------------------


class TestFlaggedStratum:
    def test_flagged_session_appears_in_flagged_stratum(self) -> None:
        state = _make_fresh_state(tick=0)
        features = np.array([[100.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
        ids = ["extreme", "a", "b", "c", "d"]
        # Mark a,b,c as flagged, extreme as not flagged
        state = _absorb_sessions(state, ids[:1], features[:1], tick_flagged=False)
        state = _absorb_sessions(state, ids[1:], features[1:], tick_flagged=True)
        scores = tuple(
            SessionScore(id=sid, d2=float(features[i, 0]), p_value=0.5)
            for i, sid in enumerate(ids)
        )
        tr = _make_tick_result()
        qcfg = QueueConfig(top_k=1, flagged=2, per_axis=0, uniform=0, seed=0)
        plan = build_review_queue(tr, scores, state, qcfg)
        # top_k gets "extreme" (d2=100 highest)
        assert "extreme" in plan.strata["top_k"]
        # flagged stratum takes highest-d2 among flagged; d2=4,3 get selected
        assert len(plan.strata["flagged"]) == 2
        assert "d" in plan.strata["flagged"] and "c" in plan.strata["flagged"]

    def test_all_candidates_flagged(self) -> None:
        state = _make_fresh_state(tick=0)
        features = np.random.default_rng(2).normal(0, 1, size=(10, 2))
        ids = [f"sess_{i}" for i in range(10)]
        state = _absorb_sessions(state, ids, features, tick_flagged=True)
        scores = tuple(
            SessionScore(id=sid, d2=float(i), p_value=0.5) for i, sid in enumerate(ids)
        )
        tr = _make_tick_result()
        qcfg = QueueConfig(top_k=3, flagged=10, per_axis=0, uniform=0, seed=0)
        plan = build_review_queue(tr, scores, state, qcfg)
        assert len(plan.strata["top_k"]) == 3
        assert len(plan.strata["flagged"]) == 7  # all remaining
        assert len(plan.unallocated) == 0


# ---------------------------------------------------------------------------
# Unit: per_axis stratum
# ---------------------------------------------------------------------------


class TestPerAxisStratum:
    def test_per_axis_selects_extreme_per_axis(self) -> None:
        state = _make_fresh_state(tick=0)
        # Create sessions with clear per-axis extremes
        features = np.array([
            [0.0, 0.0],    # a: neutral
            [10.0, 0.0],   # b: extreme on axis 0
            [-10.0, 0.0],  # c: extreme on axis 0
            [0.0, 10.0],   # d: extreme on axis 1
            [0.0, -10.0],  # e: extreme on axis 1
            [1.0, 1.0],    # f: mild
            [2.0, 2.0],    # g: mild
        ])
        ids = ["a", "b", "c", "d", "e", "f", "g"]
        state = _absorb_sessions(state, ids, features, tick_flagged=False)
        scores = tuple(
            SessionScore(id=sid, d2=0.0, p_value=0.5) for sid in ids  # equal d2 for all
        )
        tr = _make_tick_result()
        # top_k=0, flagged=0, per_axis=4 (2 per axis), uniform=0
        qcfg = QueueConfig(top_k=0, flagged=0, per_axis=4, uniform=0, seed=0)
        plan = build_review_queue(tr, scores, state, qcfg)
        per_axis_ids = set(plan.strata["per_axis"])
        assert len(per_axis_ids) == 4
        # Should include extreme sessions from both axes
        assert "b" in per_axis_ids or "c" in per_axis_ids  # extreme on axis 0
        assert "d" in per_axis_ids or "e" in per_axis_ids  # extreme on axis 1

    def test_per_axis_floor_division(self) -> None:
        """When per_axis budget is not divisible by A, floor division used."""
        state = _make_fresh_state(tick=0)
        features = np.random.default_rng(3).normal(0, 1, size=(20, 2))
        ids = [f"sess_{i}" for i in range(20)]
        state = _absorb_sessions(state, ids, features, tick_flagged=False)
        scores = tuple(
            SessionScore(id=sid, d2=float(i), p_value=0.5) for i, sid in enumerate(ids)
        )
        tr = _make_tick_result()
        # per_axis=3, A=2 -> floor(3/2)=1 per axis, total 2, 1 cascades
        qcfg = QueueConfig(top_k=0, flagged=0, per_axis=3, uniform=0, seed=0)
        plan = build_review_queue(tr, scores, state, qcfg)
        assert len(plan.strata["per_axis"]) == 2  # 1 per axis * 2 axes

    def test_per_axis_dedup_across_axes(self) -> None:
        """Same session shouldn't be selected for multiple axes in per_axis."""
        state = _make_fresh_state(tick=0)
        features = np.array([
            [100.0, 100.0],  # extreme on both axes
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ])
        ids = ["both", "a", "b", "c", "d"]
        state = _absorb_sessions(state, ids, features, tick_flagged=False)
        scores = tuple(
            SessionScore(id=sid, d2=0.0, p_value=0.5) for sid in ids
        )
        tr = _make_tick_result()
        qcfg = QueueConfig(top_k=0, flagged=0, per_axis=4, uniform=0, seed=0)
        plan = build_review_queue(tr, scores, state, qcfg)
        per_axis_ids = set(plan.strata["per_axis"])
        assert len(per_axis_ids) <= 4
        # No duplicates
        assert len(per_axis_ids) == len(plan.strata["per_axis"])


# ---------------------------------------------------------------------------
# Unit: priority ordering
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    def test_top_k_takes_priority_over_flagged(self) -> None:
        """Even flagged sessions go to top_k first if they have high d2."""
        state = _make_fresh_state(tick=0)
        features = np.array([[5.0, 0.0], [1.0, 0.0]])
        ids = ["high_d2", "low_d2"]
        # Both flagged
        state = _absorb_sessions(state, ids, features, tick_flagged=True)
        scores = (
            SessionScore(id="high_d2", d2=5.0, p_value=0.1),
            SessionScore(id="low_d2", d2=1.0, p_value=0.9),
        )
        tr = _make_tick_result()
        qcfg = QueueConfig(top_k=1, flagged=1, per_axis=0, uniform=0, seed=0)
        plan = build_review_queue(tr, scores, state, qcfg)
        assert plan.strata["top_k"] == ("high_d2",)
        assert plan.strata["flagged"] == ("low_d2",)


# ---------------------------------------------------------------------------
# Unit: d2=None handling
# ---------------------------------------------------------------------------


class TestNoneD2:
    def test_none_d2_sorted_to_bottom(self) -> None:
        """Sessions with d2=None should not appear in top_k."""
        state = _make_fresh_state(tick=0)
        features = np.array([[5.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        ids = ["good", "none_val", "ok"]
        state = _absorb_sessions(state, ids, features)
        scores = (
            SessionScore(id="good", d2=5.0, p_value=0.1),
            SessionScore(id="none_val", d2=None, p_value=None),
            SessionScore(id="ok", d2=2.0, p_value=0.5),
        )
        tr = _make_tick_result()
        qcfg = QueueConfig(top_k=2, flagged=0, per_axis=0, uniform=1, seed=0)
        plan = build_review_queue(tr, scores, state, qcfg)
        assert "good" in plan.strata["top_k"]
        assert "ok" in plan.strata["top_k"]
        # none_val should not be in top_k
        assert "none_val" not in plan.strata["top_k"]
        # none_val should be in uniform (only remaining candidate with lowest budget)
        assert "none_val" in plan.strata["uniform"]


# ---------------------------------------------------------------------------
# Hypothesis: strata invariants
# ---------------------------------------------------------------------------


class TestHypothesisStrata:
    @given(
        st.lists(
            st.tuples(
                st.text(min_size=1, max_size=12),
                st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                st.booleans(),
            ),
            min_size=1,
            max_size=100,
            unique_by=lambda t: t[0],
        ),
        st.integers(min_value=0, max_value=50),
        st.integers(min_value=0, max_value=50),
        st.integers(min_value=0, max_value=50),
        st.integers(min_value=0, max_value=50),
        st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=200)
    def test_strata_disjoint_and_budget_respected(
        self,
        rows: list[tuple[str, float, float, bool]],
        top_k: int,
        flagged: int,
        per_axis: int,
        uniform: int,
        seed: int,
    ) -> None:
        state = _make_fresh_state(tick=0)
        ids = [r[0] for r in rows]
        features = np.array([[r[1], r[2]] for r in rows])
        flagged_map = {r[0]: r[3] for r in rows}
        # Absorb flagged sessions first, then non-flagged
        flagged_ids = [sid for sid, f in flagged_map.items() if f]
        non_flagged_ids = [sid for sid, f in flagged_map.items() if not f]
        if flagged_ids:
            state = _absorb_sessions(
                state, flagged_ids,
                features[[rows.index(r) for r in rows if r[3]]],
                tick_flagged=True,
            )
        if non_flagged_ids:
            state = _absorb_sessions(
                state, non_flagged_ids,
                features[[rows.index(r) for r in rows if not r[3]]],
                tick_flagged=False,
            )

        scores = tuple(
            SessionScore(id=sid, d2=float(features[i, 0] ** 2 + features[i, 1] ** 2), p_value=0.5)
            for i, sid in enumerate(ids)
        )
        tr = _make_tick_result()
        qcfg = QueueConfig(top_k=top_k, flagged=flagged, per_axis=per_axis, uniform=uniform, seed=seed)
        plan = build_review_queue(tr, scores, state, qcfg)

        # [A1] Strata disjoint
        all_selected: set[str] = set()
        for s in ("top_k", "flagged", "per_axis", "uniform"):
            stratum_set = set(plan.strata[s])
            assert len(stratum_set & all_selected) == 0, f"{s} overlaps"
            all_selected.update(stratum_set)

        # [A1] reserved always empty
        assert plan.strata["reserved"] == ()

        # Budget adherence
        assert len(plan.strata["top_k"]) <= top_k
        # flagged + cascade from top_k can be bigger than flagged, check bounds
        assert len(plan.strata["flagged"]) <= flagged + max(0, top_k - len(plan.strata["top_k"]))
        assert len(plan.strata["per_axis"]) <= per_axis + max(
            0, top_k + flagged - len(plan.strata["top_k"]) - len(plan.strata["flagged"])
        )
        assert len(plan.strata["uniform"]) <= uniform + max(
            0, top_k + flagged + per_axis
            - len(plan.strata["top_k"]) - len(plan.strata["flagged"]) - len(plan.strata["per_axis"])
        )

        # selected + unallocated == candidate set
        assert all_selected | set(plan.unallocated) == set(ids)
        assert len(all_selected & set(plan.unallocated)) == 0


# ---------------------------------------------------------------------------
# Hypothesis: determinism
# ---------------------------------------------------------------------------


class TestHypothesisDeterminism:
    @given(
        st.lists(
            st.tuples(
                st.text(min_size=1, max_size=8),
                st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
                st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
                st.booleans(),
            ),
            min_size=5,
            max_size=50,
            unique_by=lambda t: t[0],
        ),
        st.integers(min_value=1, max_value=20),
        st.integers(min_value=1, max_value=20),
        st.integers(min_value=1, max_value=20),
        st.integers(min_value=1, max_value=20),
        st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=100)
    def test_determinism(
        self,
        rows: list[tuple[str, float, float, bool]],
        top_k: int,
        flagged: int,
        per_axis: int,
        uniform: int,
        seed: int,
    ) -> None:
        state = _make_fresh_state(tick=0)
        ids = [r[0] for r in rows]
        features = np.array([[r[1], r[2]] for r in rows])
        flagged_map = {r[0]: r[3] for r in rows}
        flagged_ids = [sid for sid, f in flagged_map.items() if f]
        non_flagged_ids = [sid for sid, f in flagged_map.items() if not f]
        if flagged_ids:
            state = _absorb_sessions(
                state, flagged_ids,
                features[[rows.index(r) for r in rows if r[3]]],
                tick_flagged=True,
            )
        if non_flagged_ids:
            state = _absorb_sessions(
                state, non_flagged_ids,
                features[[rows.index(r) for r in rows if not r[3]]],
                tick_flagged=False,
            )

        scores = tuple(
            SessionScore(id=sid, d2=float(features[i, 0] ** 2 + features[i, 1] ** 2), p_value=0.5)
            for i, sid in enumerate(ids)
        )
        tr = _make_tick_result()
        qcfg = QueueConfig(top_k=top_k, flagged=flagged, per_axis=per_axis, uniform=uniform, seed=seed)

        plan1 = build_review_queue(tr, scores, state, qcfg)
        plan2 = build_review_queue(tr, scores, state, qcfg)
        assert plan1 == plan2, f"Non-deterministic: {plan1} != {plan2}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_candidate_goes_to_top_k(self) -> None:
        state = _make_fresh_state(tick=0)
        features = np.array([[5.0, 3.0]])
        ids = ["only"]
        state = _absorb_sessions(state, ids, features)
        scores = (SessionScore(id="only", d2=34.0, p_value=0.001),)
        tr = _make_tick_result()
        qcfg = QueueConfig(top_k=5, flagged=5, per_axis=5, uniform=5, seed=0)
        plan = build_review_queue(tr, scores, state, qcfg)
        assert plan.strata["top_k"] == ("only",)
        assert plan.strata["flagged"] == ()
        assert plan.strata["per_axis"] == ()
        assert plan.strata["uniform"] == ()
        assert plan.unallocated == ()

    def test_all_candidates_same_d2(self) -> None:
        """When all have equal d2, top_k selection is stable but arbitrary."""
        state = _make_fresh_state(tick=0)
        features = np.ones((10, 2))
        ids = [f"sess_{i}" for i in range(10)]
        state = _absorb_sessions(state, ids, features)
        scores = tuple(
            SessionScore(id=sid, d2=1.0, p_value=0.5) for sid in ids
        )
        tr = _make_tick_result()
        qcfg = QueueConfig(top_k=5, flagged=0, per_axis=0, uniform=0, seed=0)
        plan = build_review_queue(tr, scores, state, qcfg)
        assert len(plan.strata["top_k"]) == 5
        assert len(plan.strata["flagged"]) == 0

    def test_candidates_without_population_match(self) -> None:
        """Candidates not in population should have zero features and not flagged."""
        state = _make_fresh_state(tick=0)
        # Don't absorb — candidates not in population
        scores = (
            SessionScore(id="ghost", d2=100.0, p_value=0.001),
            SessionScore(id="phantom", d2=50.0, p_value=0.01),
        )
        tr = _make_tick_result()
        qcfg = QueueConfig(top_k=1, flagged=5, per_axis=5, uniform=5, seed=0)
        plan = build_review_queue(tr, scores, state, qcfg)
        assert plan.strata["top_k"] == ("ghost",)
        # Phantom should go to flagged (d2=50) if flagged budget + cascade available
        # Or per_axis if not. Either way, disjoint and complete.
        all_sel = set()
        for s in ("top_k", "flagged", "per_axis", "uniform"):
            all_sel.update(plan.strata[s])
        assert all_sel | set(plan.unallocated) == {"ghost", "phantom"}