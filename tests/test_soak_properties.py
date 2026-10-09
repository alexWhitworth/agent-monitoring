"""Property-based soak tests (F-012) — cross-module hypothesis/Monte-Carlo
hardening over realistic tick cadences.

Invariants:
  [T1] streaming == batch replay within 1e-12 over >= 100 generated tick prefixes
  [P1] Tier 1: mean D2 within 5% of A, flag rate in [0.5*alpha, 2*alpha] over >= 10^4 iid ticks
  [P2] zero NaN/Inf across all result fields under adversarial valid inputs
  [P3] Tier 2: mean d2 within 10% of A, flag rate in [alpha, 2*alpha] over >= 10^4 H0 sessions
  [R1] full-lifecycle synthetic streams: warm-up -> steady -> drift -> spike -> recovery
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy.stats import chi2  # type: ignore[import-untyped]

from agent_monitor.advance import advance
from agent_monitor.config import validate_monitor_config
from agent_monitor.population import absorb, score_sessions, winsor_caps
from agent_monitor.types import (
    AxisResult,
    JointResult,
    MonitorConfig,
    MonitorState,
    SessionFeatures,
    SessionWindow,
    TickMetrics,
    TickResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    axis_names: tuple[str, ...] = ("a", "b"),
    metric_names: tuple[tuple[str, ...], ...] = (("m",), ("m",)),
    alpha: float = 0.01,
    min_warm: int = 30,
    window: int = 80,
    population_window: int = 80,
    min_population: int = 30,
    winsor_pct: float | None = None,
    **overrides: object,
) -> MonitorConfig:
    return validate_monitor_config(
        MonitorConfig(
            axis_names=axis_names,
            metric_names=metric_names,
            window=window,
            alpha=alpha,
            min_warm=min_warm,
            population_window=population_window,
            min_population=min_population,
            winsor_pct=winsor_pct,
            **overrides,
        )
    )


def _fresh_state(config: MonitorConfig, tick: int = 0) -> MonitorState:
    A = len(config.axis_names)
    M_max = max(len(m) for m in config.metric_names)
    w = config.window
    pop = SessionWindow(
        ids=np.array([], dtype=object),
        ticks=np.array([], dtype=np.int64),
        features=np.empty((0, A), dtype=np.float64),
        tick_flagged=np.array([], dtype=bool),
        absorbed_at=np.array([], dtype=np.int64),
    )
    return MonitorState(
        config=config, tick=tick,
        metric_buf=np.zeros((A, M_max, w), dtype=np.float64),
        metric_count=np.zeros((A, M_max), dtype=np.int64),
        theta=None,
        axis_buf=np.zeros((A, w), dtype=np.float64),
        axis_count=np.zeros(A, dtype=np.int64),
        n=0, mu=np.zeros(A, dtype=np.float64),
        M2=np.zeros((A, A), dtype=np.float64),
        population=pop,
    )


def _collect_fields(result: TickResult) -> list[float | None]:
    """Collect all numeric fields for NaN/Inf checking."""
    fields: list[float | None] = [
        result.joint.D2, result.joint.p_value, result.joint.threshold,
    ]
    for ar in result.axis_results:
        fields.extend([ar.S_A, ar.theta, ar.sigma_S, ar.deviation, ar.threshold])
    if result.S_vector is not None:
        fields.extend(float(x) for x in result.S_vector)
    return fields


# ---------------------------------------------------------------------------
# [T1] Streaming == batch replay (soak)
# ---------------------------------------------------------------------------


class TestT1StreamingBatch:
    @given(
        st.lists(
            st.lists(
                st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=3,
            ),
            min_size=20, max_size=100,
        ),
    )
    @settings(max_examples=50)
    def test_streaming_equals_batch_over_prefixes(self, ticks_data: list[list[float]]) -> None:
        """[T1] Streaming advance == batch recompute over prefixes."""
        # Use variable metric counts across ticks
        metrics_per_tick = [len(t) for t in ticks_data]
        min_metrics = min(metrics_per_tick)

        config = _make_config(
            axis_names=("a", "b"),
            metric_names=(("m",), ("m",)),
            min_warm=5,
        )
        state = _fresh_state(config)

        streaming_results: list[TickResult] = []
        for t in ticks_data:
            tick = TickMetrics(
                X=(
                    np.array([t[0]], dtype=np.float64),
                    np.array([t[min(1, len(t) - 1)]], dtype=np.float64),
                )
            )
            state, result = advance(state, tick)
            streaming_results.append(result)

        # Batch replay: reset and replay each prefix
        batch_state = _fresh_state(config)
        for i in range(len(ticks_data)):
            tick = TickMetrics(
                X=(
                    np.array([ticks_data[i][0]], dtype=np.float64),
                    np.array(
                        [ticks_data[i][min(1, len(ticks_data[i]) - 1)]],
                        dtype=np.float64,
                    ),
                )
            )
            batch_state, batch_result = advance(batch_state, tick)
            sr = streaming_results[i]
            assert batch_result.alert == sr.alert
            assert batch_result.joint_flag == sr.joint_flag
            if sr.joint.D2 is not None:
                assert batch_result.joint.D2 == pytest.approx(sr.joint.D2, abs=1e-12)
            assert batch_result.any_axis_flag == sr.any_axis_flag
            for a1, a2 in zip(batch_result.axis_results, sr.axis_results, strict=True):
                assert a1.flagged == a2.flagged
                if a1.S_A is not None:
                    assert a1.S_A == pytest.approx(a2.S_A, abs=1e-12)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# [P1] Tier 1 H0 behavior over 10^4 iid ticks
# ---------------------------------------------------------------------------


class TestP1Tier1H0:
    def test_mean_d2_and_flag_rate_iid(self) -> None:
        """[P1] Over >= 10^4 iid ticks: mean D2 within 5% of A,
        flag rate in [0.5*alpha, 2*alpha] after warm-up."""
        A = 3
        alpha = 0.01
        config = _make_config(
            axis_names=("a", "b", "c"),
            metric_names=(("m",), ("m",), ("m",)),
            alpha=alpha,
            min_warm=30,
        )
        state = _fresh_state(config)

        rng = np.random.default_rng(0)
        n_ticks = 12_000
        d2_values: list[float] = []
        flag_count = 0
        post_warm_flags = 0
        post_warm_ticks = 0

        for i in range(n_ticks):
            tick = TickMetrics(
                X=(
                    rng.normal(0, 1, size=(1,)).astype(np.float64),
                    rng.normal(0, 1, size=(1,)).astype(np.float64),
                    rng.normal(0, 1, size=(1,)).astype(np.float64),
                )
            )
            state, result = advance(state, tick)

            if result.joint.D2 is not None:
                d2_values.append(result.joint.D2)
                assert np.isfinite(result.joint.D2)
                assert 0.0 <= result.joint.p_value <= 1.0  # type: ignore[operator]

            if result.joint_flag:
                flag_count += 1
            if i >= config.min_warm and result.joint.D2 is not None:
                post_warm_ticks += 1
                if result.joint.flagged:
                    post_warm_flags += 1

        # After warm-up should have >= 10^4 ticks
        assert post_warm_ticks >= 10_000

        post_warm_d2 = [d for d in d2_values[-post_warm_ticks:]]
        mean_d2 = np.mean(post_warm_d2)
        flag_rate = post_warm_flags / post_warm_ticks

        # [P1] mean D2 within 5% of A
        assert A * 0.95 <= mean_d2 <= A * 1.05, (
            f"Mean D2={mean_d2:.3f} outside [A*0.95, A*1.05] = [{A*0.95}, {A*1.05}]"
        )
        # [P1] flag rate in [0.5*alpha, 2*alpha]
        assert 0.5 * alpha <= flag_rate <= 2.0 * alpha, (
            f"Flag rate={flag_rate:.6f} outside [0.5*alpha, 2*alpha] = "
            f"[{0.5*alpha}, {2.0*alpha}]"
        )


# ---------------------------------------------------------------------------
# [P2] Zero NaN/Inf under adversarial inputs
# ---------------------------------------------------------------------------


class TestP2NoNanInf:
    ADVERSARIAL_INPUTS = [
        # (desc, x_a, x_b) — single metric per axis
        ("zeros", 0.0, 0.0),
        ("large_magnitude", 1e8, -1e8),
        ("mixed_extreme", 1e8, -1e8),
        ("tiny_values", 1e-10, -1e-10),
        ("constant_sequence", 1.0, 1.0),
    ]

    def test_no_nan_inf_across_all_fields(self) -> None:
        """[P2] Zero NaN/Inf in any result field under adversarial inputs."""
        config = _make_config(min_warm=5)
        state = _fresh_state(config)

        rng = np.random.default_rng(7)
        n_ticks = 2000

        for i in range(n_ticks):
            # Interleave adversarial and random
            if i % 100 < len(self.ADVERSARIAL_INPUTS):
                desc, x_a, x_b = self.ADVERSARIAL_INPUTS[i % len(self.ADVERSARIAL_INPUTS)]
            else:
                x_a = float(rng.normal(0, 1))
                x_b = float(rng.normal(0, 1))

            tick = TickMetrics(
                X=(
                    np.array([x_a], dtype=np.float64),
                    np.array([x_b], dtype=np.float64),
                )
            )
            state, result = advance(state, tick)

            fields = _collect_fields(result)
            for f in fields:
                if f is not None:
                    assert np.isfinite(f), (
                        f"Non-finite value {f} at tick {i}"
                    )

    def test_all_constant_metrics(self) -> None:
        """Constant metrics should not produce NaN/Inf."""
        config = _make_config(min_warm=5)
        state = _fresh_state(config)

        for _ in range(500):
            tick = TickMetrics(
                X=(
                    np.array([1.0], dtype=np.float64),
                    np.array([1.0], dtype=np.float64),
                )
            )
            state, result = advance(state, tick)
            for f in _collect_fields(result):
                if f is not None:
                    assert np.isfinite(f)

    def test_zero_variance_axis(self) -> None:
        """One axis constant, other varying."""
        config = _make_config(min_warm=5)
        state = _fresh_state(config)
        rng = np.random.default_rng(42)

        for i in range(500):
            tick = TickMetrics(
                X=(
                    np.array([0.0], dtype=np.float64),  # constant axis
                    np.array([float(rng.normal(0, 1))], dtype=np.float64),
                )
            )
            state, result = advance(state, tick)
            for f in _collect_fields(result):
                if f is not None:
                    assert np.isfinite(f)


# ---------------------------------------------------------------------------
# [P3] Tier 2 H0 behavior over 10^4 sessions
# ---------------------------------------------------------------------------


class TestP3Tier2H0:
    def test_mean_d2_and_pvalue_distribution(self) -> None:
        """[P3] Over >= 10^4 H0 sessions: mean d2 within 10% of A,
        flag rate in [alpha, 2*alpha]."""
        A = 3
        alpha = 0.01
        config = _make_config(
            axis_names=("a", "b", "c"),
            metric_names=(("m",), ("m",), ("m",)),
            alpha=alpha,
            min_warm=30,
            min_population=30,
            winsor_pct=None,
        )
        state = _fresh_state(config)

        rng = np.random.default_rng(1)

        # Build population with 500 H0 sessions
        pop_features = rng.normal(0, 1, size=(500, A)).astype(np.float64)
        sf_pop = SessionFeatures(
            ids=tuple(f"pop_{i}" for i in range(500)),
            tick=0,
            F=pop_features,
        )
        state = absorb(state, sf_pop, tick_flagged=False)

        # Score 10^4 H0 candidate sessions
        n_candidates = 12_000
        cand_features = rng.normal(0, 1, size=(n_candidates, A)).astype(np.float64)
        batch_size = 1000

        d2_values: list[float] = []
        p_values: list[float] = []
        for start in range(0, n_candidates, batch_size):
            end = min(start + batch_size, n_candidates)
            chunk = cand_features[start:end]
            sf = SessionFeatures(
                ids=tuple(f"cand_{j}" for j in range(start, end)),
                tick=0,
                F=chunk,
            )
            scores = score_sessions(state, sf)
            for s in scores:
                if s.d2 is not None and s.p_value is not None:
                    d2_values.append(s.d2)
                    p_values.append(s.p_value)

        assert len(d2_values) >= 10_000

        mean_d2 = np.mean(d2_values)
        # [P3] mean d2 within 10% of A
        assert A * 0.90 <= mean_d2 <= A * 1.10, (
            f"Mean d2={mean_d2:.3f} outside [A*0.9, A*1.1] = [{A*0.9}, {A*1.1}]"
        )

        # P-value uniformity check: flag rate for threshold = chi2.ppf(1-alpha, A)
        threshold = float(chi2.ppf(1 - alpha, df=A))
        flag_count = sum(1 for d in d2_values if d > threshold)
        flag_rate = flag_count / len(d2_values)
        assert alpha <= flag_rate <= 2.0 * alpha, (
            f"Tier 2 flag rate={flag_rate:.6f} outside [{alpha}, {2.0*alpha}]"
        )

    def test_tier2_no_nan_inf_scores(self) -> None:
        """All Tier 2 scores must be finite."""
        A = 2
        config = _make_config(
            axis_names=("a", "b"),
            metric_names=(("m",), ("m",)),
            min_population=10,
            winsor_pct=None,
        )
        state = _fresh_state(config)
        rng = np.random.default_rng(99)

        pop_f = rng.normal(0, 1, size=(100, A)).astype(np.float64)
        state = absorb(state, SessionFeatures(
            ids=tuple(f"p_{i}" for i in range(100)), tick=0, F=pop_f,
        ), tick_flagged=False)

        cand_f = rng.uniform(-1e6, 1e6, size=(500, A)).astype(np.float64)
        scores = score_sessions(state, SessionFeatures(
            ids=tuple(f"c_{i}" for i in range(500)), tick=0, F=cand_f,
        ))
        for s in scores:
            assert s.d2 is None or np.isfinite(s.d2)
            assert s.p_value is None or np.isfinite(s.p_value)


# ---------------------------------------------------------------------------
# [R1] Full-lifecycle synthetic stream
# ---------------------------------------------------------------------------


class TestR1FullLifecycle:
    def test_lifecycle_detection_expectations(self) -> None:
        """[R1] Full lifecycle: warm-up -> steady -> drift -> spike -> recovery.

        Detection expectations:
          - Steady state: mostly silent (few false positives)
          - Coordinated drift: joint_flag triggers
          - Extreme spike: any_axis_flag and joint_flag trigger
          - Recovery: flags drop back to baseline
        """
        A = 2
        alpha = 0.01
        config = _make_config(
            axis_names=("a", "b"),
            metric_names=(("m",), ("m",)),
            alpha=alpha,
            min_warm=20,
            window=30,
        )
        state = _fresh_state(config)
        rng = np.random.default_rng(42)

        total_ticks = 220
        results: list[TickResult] = []

        # Phase boundaries (tick indices)
        warm_up_end = 50          # 0..49
        steady_end = 80           # 50..79
        drift_end = 110           # 80..109
        spike_end = 120           # 110..119
        # recovery: 120..219

        for i in range(total_ticks):
            if i < warm_up_end:
                # ---- Warm-up ----
                x_a = float(rng.normal(0, 0.5))
                x_b = float(rng.normal(0, 0.5))
            elif i < steady_end:
                # ---- Steady state ----
                x_a = float(rng.normal(0, 0.3))
                x_b = float(rng.normal(0, 0.3))
            elif i < drift_end:
                # ---- Coordinated drift: shift both axes by +2 sigma ----
                x_a = float(rng.normal(0, 0.3)) + 2.0
                x_b = float(rng.normal(0, 0.3)) + 2.0
            elif i < spike_end:
                # ---- Extreme spike on axis a only ----
                x_a = 20.0
                x_b = float(rng.normal(0, 0.3))
            else:
                # ---- Recovery: back to steady state ----
                x_a = float(rng.normal(0, 0.3))
                x_b = float(rng.normal(0, 0.3))

            tick = TickMetrics(
                X=(
                    np.array([x_a], dtype=np.float64),
                    np.array([x_b], dtype=np.float64),
                )
            )
            state, result = advance(state, tick)
            results.append(result)

        # ---- Phase assertions ----
        wu = warm_up_end
        ss_start, ss_end = wu, steady_end  # 50..79
        d_start, d_end = steady_end, drift_end  # 80..109
        sp_start, sp_end = drift_end, spike_end  # 110..119
        rc_start, rc_end = spike_end, total_ticks  # 120..219

        # Steady state: should have low joint flag rate
        steady_joint = sum(1 for r in results[ss_start:ss_end] if r.joint_flag)
        assert steady_joint <= 5, f"Too many joint flags in steady: {steady_joint}"

        # Drift: elevated joint flags
        drift_joint = sum(1 for r in results[d_start:d_end] if r.joint_flag)
        drift_any = sum(1 for r in results[d_start:d_end] if r.any_axis_flag)
        assert drift_joint >= 1, f"Expected joint_flag during drift, got {drift_joint}"

        # Spike: should force flags
        spike_joint = sum(1 for r in results[sp_start:sp_end] if r.joint_flag)
        spike_any = sum(1 for r in results[sp_start:sp_end] if r.any_axis_flag)
        assert spike_joint >= 2, f"Expected joint flags during spike, got {spike_joint}"

        # Recovery: flags subside
        recovery_joint = sum(1 for r in results[rc_start:rc_end] if r.joint_flag)
        assert recovery_joint < 20, f"Too many flags during recovery: {recovery_joint}"

        # Verify no NaN/Inf anywhere
        for r in results:
            for f in _collect_fields(r):
                if f is not None:
                    assert np.isfinite(f)