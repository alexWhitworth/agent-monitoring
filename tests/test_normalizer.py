"""Tests for normalizer.py — rolling z-score (API-004)."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from agent_monitor.normalizer import zscore


def _recompute_zscore(buf: np.ndarray, count: int, x: float) -> float | None:
    """Oracle: recompute z-score from buffer contents.

    count is the total number of inserts (unbounded).
    """
    w = len(buf)
    if count < 2:
        return None
    pos = (count - 1) % w
    if count <= w:
        valid = buf[:count]
    else:
        start = (pos + 1) % w
        if start == 0:
            valid = buf
        else:
            valid = np.concatenate([buf[start:], buf[:start]])
    mu = float(np.mean(valid))
    sigma = float(np.std(valid, ddof=1))
    if sigma < 1e-10:
        return 0.0
    return (x - mu) / sigma


# ---------------------------------------------------------------------------
# Unit: warm-up
# ---------------------------------------------------------------------------


class TestWarmUp:
    def test_count_zero_returns_none(self) -> None:
        buf = np.zeros(10, dtype=np.float64)
        buf.flags.writeable = False
        z, _new_buf, count = zscore(buf, 0, 5.0)
        assert z is None
        assert count == 1

    def test_count_one_yields_valid_z(self) -> None:
        """With 1 prior sample, adding a second yields a valid z-score (>=2 samples)."""
        buf = np.array([3.0] + [0.0] * 9, dtype=np.float64)
        buf.flags.writeable = False
        z, _new_buf, count = zscore(buf, 1, 5.0)
        assert isinstance(z, float)
        assert count == 2
        # z-score of 5.0 given [3.0, 5.0]: mu=4.0, sigma=std([3,5],ddof=1)=1.4142
        # z = (5-4)/1.4142 = 0.707
        expected = (5.0 - 4.0) / np.std([3.0, 5.0], ddof=1)
        assert z == pytest.approx(float(expected), rel=0.0, abs=1e-12)

    def test_count_two_adds_third(self) -> None:
        """With 2 prior samples, adding a third computes z from 3 samples."""
        buf = np.array([3.0, 0.0] + [0.0] * 8, dtype=np.float64)
        buf.flags.writeable = False
        z, _new_buf, count = zscore(buf, 2, 5.0)
        assert isinstance(z, float)
        assert count == 3
        expected = (5.0 - 8.0 / 3.0) / np.std([3.0, 0.0, 5.0], ddof=1)
        assert z == pytest.approx(float(expected), rel=0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Unit: constant metric guard
# ---------------------------------------------------------------------------


class TestConstantMetric:
    def test_constant_metric_gives_zero_zscore(self) -> None:
        buf = np.zeros(10, dtype=np.float64)
        buf.flags.writeable = False
        c = 0
        b = buf
        for _i in range(10):
            z, b, c = zscore(b, c, 42.0)
        # After 10 inserts, the last z should be 0.0
        assert z == 0.0

    def test_near_constant_gives_zero_zscore(self) -> None:
        """sigma < 1e-10 triggers z=0.0."""
        buf = np.zeros(10, dtype=np.float64)
        buf.flags.writeable = False
        c = 0
        b = buf
        vals = [1.0, 1.0 + 1e-11, 1.0 - 1e-11] * 3 + [1.0]
        for v in vals:
            z, b, c = zscore(b, c, v)
        assert z == 0.0


# ---------------------------------------------------------------------------
# Unit: eviction
# ---------------------------------------------------------------------------


class TestEviction:
    def test_ring_buffer_eviction(self) -> None:
        """After w+5 updates, buffer contains exactly the last w samples in order."""
        w = 10
        buf = np.zeros(w, dtype=np.float64)
        buf.flags.writeable = False
        c = 0
        b = buf
        # Feed 15 values: 1..15
        for v in range(1, 16):
            _z, b, c = zscore(b, c, float(v))
        # c should be 15 (total inserts)
        assert c == 15
        # Reconstruct the valid window: entries 6..15 (last w=10)
        # They were written at positions: 5%10=5, 6,7,8,9,0,1,2,3,4
        # So the valid window in ring order starts at pos+1 = (14%10+1)%10 = 5
        expected = np.array([6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        # Reconstruct from buffer
        pos = (c - 1) % w  # last write position = 14 % 10 = 4
        start = (pos + 1) % w  # = 5
        reconstructed = np.concatenate([b[start:], b[:start]])
        np.testing.assert_array_equal(reconstructed, expected)

    def test_ring_buffer_contents_after_wrap(self) -> None:
        """After exactly w inserts, buffer is full and in insertion order."""
        w = 5
        buf = np.zeros(w, dtype=np.float64)
        buf.flags.writeable = False
        c = 0
        b = buf
        for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
            _z, b, c = zscore(b, c, v)
        assert c == 5
        # Buffer should be [10,20,30,40,50] since no wrap occurred
        np.testing.assert_array_equal(b, np.array([10.0, 20.0, 30.0, 40.0, 50.0]))


# ---------------------------------------------------------------------------
# Unit: basic oracle match
# ---------------------------------------------------------------------------


class TestBasicOracle:
    def test_oracle_match_simple_sequence(self) -> None:
        """zscore result matches recomputing from the buffer directly."""
        w = 10
        buf = np.zeros(w, dtype=np.float64)
        buf.flags.writeable = False
        c = 0
        b = buf
        for v in [1.0, 3.0, 5.0, 7.0, 9.0]:
            z, b, c = zscore(b, c, v)
            expected = _recompute_zscore(b, c, v)
            if expected is None:
                assert z is None
            else:
                assert z == pytest.approx(expected, rel=0.0, abs=1e-12)

    def test_oracle_match_across_wrap_boundary(self) -> None:
        """zscore still matches oracle after buffer wraps."""
        w = 5
        buf = np.zeros(w, dtype=np.float64)
        buf.flags.writeable = False
        c = 0
        b = buf
        rng = np.random.default_rng(123)
        for _ in range(20):
            v = float(rng.normal(0, 1))
            z, b, c = zscore(b, c, v)
            expected = _recompute_zscore(b, c, v)
            if expected is None:
                assert z is None
            else:
                assert z == pytest.approx(expected, rel=0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Hypothesis: oracle match
# ---------------------------------------------------------------------------

_finite_float = st.floats(
    allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6
)
_stream = st.lists(_finite_float, min_size=2, max_size=100)


class TestHypothesisOracle:
    @given(_stream)
    def test_zscore_matches_recomputed_oracle(self, stream: list[float]) -> None:
        w = 20
        buf = np.zeros(w, dtype=np.float64)
        buf.flags.writeable = False
        c = 0
        b = buf
        for x in stream:
            z, b, c = zscore(b, c, x)
            expected = _recompute_zscore(b, c, x)
            if expected is None:
                assert z is None
            else:
                assert z == pytest.approx(expected, rel=0.0, abs=1e-12)


class TestHypothesisIID:
    @given(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False, min_value=-10.0, max_value=10.0),
            min_size=80,
            max_size=120,
        )
    )
    def test_filled_window_outputs_finite(self, stream: list[float]) -> None:
        """After filling the window, z-scores are finite."""
        w = 80
        buf = np.zeros(w, dtype=np.float64)
        buf.flags.writeable = False
        c = 0
        b = buf
        for i, x in enumerate(stream):
            z, b, c = zscore(b, c, x)
            if i >= 79 and z is not None:
                assert math.isfinite(z)
