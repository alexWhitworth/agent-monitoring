"""Tests for types.py and config.py - MonitorConfig, QueueConfig, JSON codec."""

import json

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.stats import chi2  # type: ignore[import-untyped]

from agent_monitor.config import (
    VALID_AGG_NAMES,
    config_from_json,
    config_to_json,
    validate_monitor_config,
    validate_queue_config,
)
from agent_monitor.types import (
    AxisResult,
    JointResult,
    MonitorConfig,
    QueueConfig,
    ReviewQueuePlan,
    SessionFeatures,
    SessionScore,
    SessionWindow,
    TickMetrics,
    TickResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_valid_config(**overrides: object) -> MonitorConfig:
    """Build a valid MonitorConfig with paper defaults, optionally overridden."""
    defaults: dict[str, object] = {
        "axis_names": ("Safety", "Quality", "Efficiency"),
        "metric_names": (("m1", "m2"), ("m1",), ("m1", "m2", "m3")),
        "window": 80,
        "lam": 0.25,
        "k": None,
        "alpha": 0.01,
        "min_warm": 30,
        "ridge": 1e-6,
        "agg": "mean",
        "population_window": 80,
        "min_population": 30,
        "winsor_pct": 0.99,
        "winsor_k": 3,
    }
    defaults.update(overrides)
    return MonitorConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Dataclass construction / immutability
# ---------------------------------------------------------------------------


class TestDataclassImmutability:
    """All schemas from DS-001..DS-011 are frozen."""

    def test_monitor_config_is_frozen(self) -> None:
        cfg = make_valid_config()
        with pytest.raises(Exception):  # noqa: B017
            cfg.window = 99  # type: ignore[misc]

    def test_queue_config_is_frozen(self) -> None:
        q = QueueConfig()
        with pytest.raises(Exception):  # noqa: B017
            q.top_k = 99  # type: ignore[misc]

    def test_axis_result_is_frozen(self) -> None:
        r = AxisResult("Safety", 1.0, 0.5, 0.3, 0.5, 4.5, False, "ok")
        with pytest.raises(Exception):  # noqa: B017
            r.flagged = True  # type: ignore[misc]

    def test_joint_result_is_frozen(self) -> None:
        r = JointResult(30, 5.0, 15.0, 0.01, False, "ok")
        with pytest.raises(Exception):  # noqa: B017
            r.flagged = True  # type: ignore[misc]

    def test_tick_result_is_frozen(self) -> None:
        ar = AxisResult("Safety", 1.0, 0.5, 0.3, 0.5, 4.5, False, "ok")
        jr = JointResult(30, 5.0, 15.0, 0.01, False, "ok")
        tr = TickResult(0, (ar,), jr, np.array([1.0]), False, False, False)
        with pytest.raises(Exception):  # noqa: B017
            tr.tick = 1  # type: ignore[misc]

    def test_session_score_is_frozen(self) -> None:
        s = SessionScore("s1", 3.0, 0.05)
        with pytest.raises(Exception):  # noqa: B017
            s.d2 = None  # type: ignore[misc]

    def test_review_queue_plan_is_frozen(self) -> None:
        rp = ReviewQueuePlan(0, {"top_k": ("s1",), "reserved": ()}, ("s2",))
        with pytest.raises(Exception):  # noqa: B017
            rp.tick = 1  # type: ignore[misc]


class TestNumPyArraysReadOnly:
    """All NumPy arrays in immutable schemas are write-protected."""

    def test_session_window_arrays_read_only(self) -> None:
        sw = SessionWindow(
            ids=np.array(["a", "b"]),
            ticks=np.array([0, 1], dtype=np.int64),
            features=np.array([[1.0, 2.0], [3.0, 4.0]]),
            tick_flagged=np.array([False, True]),
            absorbed_at=np.array([0, 1], dtype=np.int64),
        )
        assert not sw.features.flags.writeable

    def test_tick_metrics_arrays_read_only(self) -> None:
        tm = TickMetrics(X=(np.array([1.0, 2.0]), np.array([3.0])))
        for arr in tm.X:
            assert not arr.flags.writeable

    def test_session_features_array_read_only(self) -> None:
        sf = SessionFeatures(ids=("a", "b"), tick=0, F=np.array([[1.0], [2.0]]))
        assert not sf.F.flags.writeable


# ---------------------------------------------------------------------------
# validate_monitor_config — acceptance cases
# ---------------------------------------------------------------------------


class TestValidateMonitorConfigAccept:
    def test_valid_default_config(self) -> None:
        cfg = validate_monitor_config(make_valid_config())
        # k should be resolved
        assert cfg.k == pytest.approx(chi2.ppf(0.99, df=3))

    def test_k_none_resolves_to_chi2_per_axis_count(self) -> None:
        for num_axes in (2, 3, 5, 7):
            cfg = make_valid_config(
                axis_names=tuple(f"a{i}" for i in range(num_axes)),
                metric_names=tuple(("m",) for _ in range(num_axes)),
            )
            result = validate_monitor_config(cfg)
            assert result.k == pytest.approx(float(chi2.ppf(0.99, df=num_axes)))

    def test_k_provided_is_preserved(self) -> None:
        cfg = make_valid_config(k=10.0)
        result = validate_monitor_config(cfg)
        assert result.k == 10.0

    def test_k_exact_chi2_5(self) -> None:
        """k=None with A=5 resolves to chi2_5(0.99) ~ 15.0863, within 1e-9."""
        cfg = make_valid_config(
            axis_names=("a", "b", "c", "d", "e"),
            metric_names=(("m",),) * 5,
        )
        result = validate_monitor_config(cfg)
        expected = float(chi2.ppf(0.99, df=5))
        assert result.k == pytest.approx(expected, rel=0.0, abs=1e-9)
        # Numerical sanity: expected within [15.086, 15.087]
        assert 15.086 <= result.k <= 15.087

    def test_minimal_valid(self) -> None:
        """Smallest valid config: 2 axes, 1 metric each, minimal window/warm."""
        cfg = make_valid_config(
            axis_names=("a", "b"),
            metric_names=(("m",), ("m",)),
            window=2,
            min_warm=2,
            population_window=1,
            winsor_k=1,
            winsor_pct=None,
            k=1.0,
        )
        result = validate_monitor_config(cfg)
        assert result.k == 1.0

    def test_winsor_pct_none_accepted(self) -> None:
        cfg = make_valid_config(winsor_pct=None)
        result = validate_monitor_config(cfg)
        assert result.winsor_pct is None

    def test_each_agg_name_accepted(self) -> None:
        for agg in sorted(VALID_AGG_NAMES):
            cfg = make_valid_config(agg=agg)
            result = validate_monitor_config(cfg)
            assert result.agg == agg


# ---------------------------------------------------------------------------
# validate_monitor_config — rejection cases (every rule)
# ---------------------------------------------------------------------------


class TestValidateMonitorConfigReject:
    def test_less_than_2_axes(self) -> None:
        cfg = make_valid_config(axis_names=("a",), metric_names=(("m",),))
        with pytest.raises(ValueError, match="At least 2 axes"):
            validate_monitor_config(cfg)

    def test_metric_names_length_mismatch(self) -> None:
        cfg = make_valid_config(
            axis_names=("a", "b", "c"), metric_names=(("m",), ("m",))
        )
        with pytest.raises(ValueError, match="metric_names length"):
            validate_monitor_config(cfg)

    def test_axis_with_zero_metrics(self) -> None:
        cfg = make_valid_config(
            axis_names=("a", "b"), metric_names=(("m",), ())
        )
        with pytest.raises(ValueError, match=">= 1 metric"):
            validate_monitor_config(cfg)

    def test_window_too_small(self) -> None:
        cfg = make_valid_config(window=1)
        with pytest.raises(ValueError, match="window must be >= 2"):
            validate_monitor_config(cfg)

    def test_window_zero(self) -> None:
        cfg = make_valid_config(window=0)
        with pytest.raises(ValueError, match="window must be >= 2"):
            validate_monitor_config(cfg)

    def test_lam_zero(self) -> None:
        cfg = make_valid_config(lam=0.0)
        with pytest.raises(ValueError, match="lam must be in"):
            validate_monitor_config(cfg)

    def test_lam_above_one(self) -> None:
        cfg = make_valid_config(lam=1.5)
        with pytest.raises(ValueError, match="lam must be in"):
            validate_monitor_config(cfg)

    def test_lam_negative(self) -> None:
        cfg = make_valid_config(lam=-0.1)
        with pytest.raises(ValueError, match="lam must be in"):
            validate_monitor_config(cfg)

    def test_min_warm_too_small(self) -> None:
        cfg = make_valid_config(min_warm=1)
        with pytest.raises(ValueError, match="min_warm must be >= 2"):
            validate_monitor_config(cfg)

    def test_population_window_zero(self) -> None:
        cfg = make_valid_config(population_window=0)
        with pytest.raises(ValueError, match="population_window must be >= 1"):
            validate_monitor_config(cfg)

    def test_winsor_k_zero(self) -> None:
        cfg = make_valid_config(winsor_k=0)
        with pytest.raises(ValueError, match="winsor_k must be >= 1"):
            validate_monitor_config(cfg)

    def test_winsor_pct_zero(self) -> None:
        cfg = make_valid_config(winsor_pct=0.0)
        with pytest.raises(ValueError, match="winsor_pct must be None or in"):
            validate_monitor_config(cfg)

    def test_winsor_pct_one(self) -> None:
        cfg = make_valid_config(winsor_pct=1.0)
        with pytest.raises(ValueError, match="winsor_pct must be None or in"):
            validate_monitor_config(cfg)

    def test_winsor_pct_negative(self) -> None:
        cfg = make_valid_config(winsor_pct=-0.5)
        with pytest.raises(ValueError, match="winsor_pct must be None or in"):
            validate_monitor_config(cfg)

    def test_winsor_pct_above_one(self) -> None:
        cfg = make_valid_config(winsor_pct=1.5)
        with pytest.raises(ValueError, match="winsor_pct must be None or in"):
            validate_monitor_config(cfg)

    def test_unknown_agg(self) -> None:
        cfg = make_valid_config(agg="median")
        with pytest.raises(ValueError, match="Unknown aggregation"):
            validate_monitor_config(cfg)

    def test_k_zero(self) -> None:
        cfg = make_valid_config(k=0.0)
        with pytest.raises(ValueError, match="k must be > 0"):
            validate_monitor_config(cfg)

    def test_k_negative(self) -> None:
        cfg = make_valid_config(k=-5.0)
        with pytest.raises(ValueError, match="k must be > 0"):
            validate_monitor_config(cfg)


# ---------------------------------------------------------------------------
# validate_queue_config
# ---------------------------------------------------------------------------


class TestValidateQueueConfig:
    def test_default_valid(self) -> None:
        q = QueueConfig()
        result = validate_queue_config(q)
        assert result is q

    def test_all_zero_valid(self) -> None:
        q = QueueConfig(top_k=0, flagged=0, per_axis=0, uniform=0, reserved=0)
        validate_queue_config(q)

    def test_negative_top_k(self) -> None:
        q = QueueConfig(top_k=-1)
        with pytest.raises(ValueError, match="top_k must be >= 0"):
            validate_queue_config(q)

    def test_negative_flagged(self) -> None:
        q = QueueConfig(flagged=-1)
        with pytest.raises(ValueError, match="flagged must be >= 0"):
            validate_queue_config(q)

    def test_negative_per_axis(self) -> None:
        q = QueueConfig(per_axis=-1)
        with pytest.raises(ValueError, match="per_axis must be >= 0"):
            validate_queue_config(q)

    def test_negative_uniform(self) -> None:
        q = QueueConfig(uniform=-1)
        with pytest.raises(ValueError, match="uniform must be >= 0"):
            validate_queue_config(q)

    def test_negative_reserved(self) -> None:
        q = QueueConfig(reserved=-1)
        with pytest.raises(ValueError, match="reserved must be >= 0"):
            validate_queue_config(q)


# ---------------------------------------------------------------------------
# JSON codec
# ---------------------------------------------------------------------------


class TestJSONCodec:
    def test_round_trip_default_config(self) -> None:
        cfg = validate_monitor_config(make_valid_config())
        json_str = config_to_json(cfg)
        cfg2 = config_from_json(json_str)
        assert cfg == cfg2

    def test_round_trip_custom_config(self) -> None:
        cfg = validate_monitor_config(
            make_valid_config(k=10.0, winsor_pct=None, agg="max", lam=0.5, window=40)
        )
        json_str = config_to_json(cfg)
        cfg2 = config_from_json(json_str)
        assert cfg == cfg2

    def test_round_trip_includes_all_fields(self) -> None:
        cfg = validate_monitor_config(make_valid_config())
        d = json.loads(config_to_json(cfg))
        expected_keys = {
            "axis_names", "metric_names", "window", "lam", "k", "alpha",
            "min_warm", "ridge", "agg", "population_window", "min_population",
            "winsor_pct", "winsor_k",
        }
        assert set(d.keys()) == expected_keys

    def test_from_json_detects_malformed(self) -> None:
        with pytest.raises(ValueError, match="Malformed config JSON"):
            config_from_json("not json{")

    def test_from_json_rejects_non_object(self) -> None:
        with pytest.raises(ValueError, match="JSON object"):
            config_from_json("[1, 2, 3]")

    def test_from_json_rejects_missing_keys(self) -> None:
        with pytest.raises(ValueError, match="Missing config keys"):
            config_from_json('{"window": 80}')

    def test_from_json_rejects_unknown_agg(self) -> None:
        cfg = make_valid_config()
        json_str = config_to_json(cfg)
        # Munge the agg field
        d = json.loads(json_str)
        d["agg"] = "median"
        with pytest.raises(ValueError, match="Unknown aggregation"):
            config_from_json(json.dumps(d))

    def test_from_json_rejects_bad_agg_type(self) -> None:
        cfg = make_valid_config()
        json_str = config_to_json(cfg)
        d = json.loads(json_str)
        d["agg"] = 42
        with pytest.raises(ValueError, match="Unknown aggregation"):
            config_from_json(json.dumps(d))


# ---------------------------------------------------------------------------
# Hypothesis: valid config generation round-trips
# ---------------------------------------------------------------------------

_axis_name = st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz")
_metric_name = st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz_")

_valid_config_strategy = st.builds(
    make_valid_config,
    axis_names=st.lists(_axis_name, min_size=2, max_size=5, unique=True).map(tuple),
    metric_names=st.lists(
        st.lists(_metric_name, min_size=1, max_size=5).map(tuple),
        min_size=2,
        max_size=5,
    ).map(tuple),
    window=st.integers(min_value=2, max_value=200),
    lam=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    k=st.one_of(st.none(), st.floats(min_value=0.01, max_value=100.0)),
    alpha=st.floats(min_value=0.001, max_value=0.5),
    min_warm=st.integers(min_value=2, max_value=200),
    ridge=st.floats(min_value=1e-12, max_value=1e-3),
    agg=st.sampled_from(sorted(VALID_AGG_NAMES)),
    population_window=st.integers(min_value=1, max_value=500),
    min_population=st.integers(min_value=1, max_value=200),
    winsor_pct=st.one_of(
        st.none(),
        st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    ),
    winsor_k=st.integers(min_value=1, max_value=20),
).filter(
    # Ensure metric_names matches axis_names length
    lambda cfg: len(cfg.metric_names) == len(cfg.axis_names)
)


class TestHypothesisValidConfig:
    @given(_valid_config_strategy)
    def test_valid_config_round_trips_through_json(self, cfg: MonitorConfig) -> None:
        validated = validate_monitor_config(cfg)
        json_str = config_to_json(validated)
        restored = config_from_json(json_str)
        assert validated == restored

    @given(_valid_config_strategy)
    def test_valid_config_validation_passes(self, cfg: MonitorConfig) -> None:
        result = validate_monitor_config(cfg)
        assert result.k is not None
        assert result.k > 0

    @given(_valid_config_strategy)
    def test_k_none_config_resolves(self, cfg: MonitorConfig) -> None:
        """When k=None, resolved value matches scipy chi2.ppf."""
        if cfg.k is not None:
            return  # skip: this test only applies to k=None
        # Create a fresh config with k forced to None
        cfg_none = make_valid_config(
            axis_names=cfg.axis_names,
            metric_names=cfg.metric_names,
            window=cfg.window,
            lam=cfg.lam,
            k=None,
            alpha=cfg.alpha,
            min_warm=cfg.min_warm,
            ridge=cfg.ridge,
            agg=cfg.agg,
            population_window=cfg.population_window,
            min_population=cfg.min_population,
            winsor_pct=cfg.winsor_pct,
            winsor_k=cfg.winsor_k,
        )
        result = validate_monitor_config(cfg_none)
        expected = float(chi2.ppf(0.99, df=len(cfg_none.axis_names)))
        assert result.k == pytest.approx(expected, rel=0.0, abs=1e-9)


# Boundary violation cases are exhaustively covered by
# TestValidateMonitorConfigReject (one explicit accept+reject per rule).
