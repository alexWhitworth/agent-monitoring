"""Tests for agg.py — aggregation registry (API-003)."""

import math
import statistics

import pytest
from hypothesis import given
from hypothesis import strategies as st

from agent_monitor.agg import AGGREGATORS, get_agg

# ---------------------------------------------------------------------------
# Oracle helpers
# ---------------------------------------------------------------------------


def _oracle_mean(zs: list[float]) -> float:
    return statistics.fmean(zs)


def _oracle_max(zs: list[float]) -> float:
    return max(zs)


def _oracle_l2(zs: list[float]) -> float:
    return math.sqrt(sum(z * z for z in zs))


# ---------------------------------------------------------------------------
# Unit: known names, empty input, unknown name
# ---------------------------------------------------------------------------


class TestRegistryLookup:
    def test_get_mean_returns_agg_function(self) -> None:
        fn = get_agg("mean")
        assert callable(fn)
        assert fn([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_get_max_returns_agg_function(self) -> None:
        fn = get_agg("max")
        assert callable(fn)
        assert fn([1.0, -5.0, 3.0]) == pytest.approx(3.0)

    def test_get_l2_returns_agg_function(self) -> None:
        fn = get_agg("l2")
        assert callable(fn)
        assert fn([3.0, 4.0]) == pytest.approx(5.0)

    def test_unknown_name_raises_keyerror(self) -> None:
        with pytest.raises(KeyError, match="Unknown aggregation 'bad'"):
            get_agg("bad")

    def test_all_registry_keys_match_valid_names(self) -> None:
        from agent_monitor.config import VALID_AGG_NAMES

        assert set(AGGREGATORS.keys()) == VALID_AGG_NAMES


class TestEmptyInput:
    """All three aggregators raise ValueError on empty input."""

    def test_mean_empty(self) -> None:
        fn = get_agg("mean")
        with pytest.raises(ValueError, match="requires at least one z-score"):
            fn([])

    def test_max_empty(self) -> None:
        fn = get_agg("max")
        with pytest.raises(ValueError, match="requires at least one z-score"):
            fn([])

    def test_l2_empty(self) -> None:
        fn = get_agg("l2")
        with pytest.raises(ValueError, match="requires at least one z-score"):
            fn([])


class TestUnitBoundaries:
    """Edge cases: single element, mixed sign, plausibility bounds."""

    def test_mean_single_element(self) -> None:
        fn = get_agg("mean")
        assert fn([42.0]) == pytest.approx(42.0)

    def test_max_single_element(self) -> None:
        fn = get_agg("max")
        assert fn([-3.14]) == pytest.approx(-3.14)

    def test_max_all_negative(self) -> None:
        fn = get_agg("max")
        assert fn([-10.0, -5.0, -1.0]) == pytest.approx(-1.0)

    def test_l2_single_element(self) -> None:
        fn = get_agg("l2")
        assert fn([5.0]) == pytest.approx(5.0)

    def test_mean_mixed_sign(self) -> None:
        fn = get_agg("mean")
        result = fn([-10.0, 0.0, 10.0])
        assert result == pytest.approx(0.0)

    def test_l2_zero_elements(self) -> None:
        fn = get_agg("l2")
        assert fn([0.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_mean_bounded_by_range(self) -> None:
        """mean(z) in [min(z), max(z)] for all inputs."""
        fn = get_agg("mean")
        zs = [-100.0, 50.0, 200.0]
        result = fn(zs)
        assert min(zs) <= result <= max(zs)

    def test_l2_non_negative(self) -> None:
        """l2 always returns non-negative."""
        fn = get_agg("l2")
        assert fn([-3.0, -4.0]) == pytest.approx(5.0)

    def test_max_returns_largest(self) -> None:
        """max returns the numerically largest element."""
        fn = get_agg("max")
        zs = [1.0, -2.5, 1.3]
        result = fn(zs)
        assert result == pytest.approx(1.3)


# ---------------------------------------------------------------------------
# Hypothesis: oracle comparison
# ---------------------------------------------------------------------------


_finite_float = st.floats(
    allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6
)
_zs_seq = st.lists(_finite_float, min_size=1, max_size=50)


class TestHypothesisOracle:
    @given(_zs_seq)
    def test_mean_matches_oracle(self, zs: list[float]) -> None:
        fn = get_agg("mean")
        assert fn(zs) == pytest.approx(_oracle_mean(zs), rel=0.0, abs=1e-12)

    @given(_zs_seq)
    def test_max_matches_oracle(self, zs: list[float]) -> None:
        fn = get_agg("max")
        assert fn(zs) == pytest.approx(_oracle_max(zs), rel=0.0, abs=1e-12)

    @given(_zs_seq)
    def test_l2_matches_oracle(self, zs: list[float]) -> None:
        fn = get_agg("l2")
        assert fn(zs) == pytest.approx(_oracle_l2(zs), rel=0.0, abs=1e-12)

    @given(_zs_seq)
    def test_mean_within_bounds(self, zs: list[float]) -> None:
        fn = get_agg("mean")
        result = fn(zs)
        # Due to fsum + float division rounding, result may exceed max by ~1 ULP
        assert min(zs) - 1e-10 <= result <= max(zs) + 1e-10

    @given(_zs_seq)
    def test_outputs_are_finite(self, zs: list[float]) -> None:
        for name in ("mean", "max", "l2"):
            fn = get_agg(name)
            result = fn(zs)
            assert math.isfinite(result)
