"""Rolling z-score normalizer for a single metric (Step 1).

API-004: zscore(buf, count, x) -> (z, buf', count')
Ring-buffer semantics with constant-metric guard and warm-up None semantics.

count tracks the total number of inserts (unbounded); the number of valid
samples is min(count, w).  The write position is (count % w).
"""

from __future__ import annotations

__all__ = ["zscore"]

import numpy as np


def zscore(
    buf: np.ndarray, count: int, x: float
) -> tuple[float | None, np.ndarray, int]:
    """Append x to one metric's ring buffer, return (z, new_buffer, new_count).

    Args:
        buf: (w,) float64 read-only ring buffer.
        count: Total number of inserts so far (unbounded int).
        x: Finite raw metric reading.

    Returns:
        z: z-score or None (warm-up: fewer than 2 total inserts).
        buf: Updated ring buffer (new array, writeable=False).
        count: Updated insert count (count + 1).
    """
    w = len(buf)
    pos = count % w
    new_buf = np.copy(buf)
    new_buf[pos] = x
    new_count = count + 1
    new_buf.flags.writeable = False

    if new_count < 2:
        return None, new_buf, new_count

    # Reconstruct valid entries in insertion order
    if new_count <= w:
        # Buffer not yet full; valid entries are positions 0..new_count-1
        valid = new_buf[:new_count]
    else:
        # Buffer full; ring order starts at (pos + 1) % w
        start = (pos + 1) % w
        if start == 0:
            valid = new_buf
        else:
            valid = np.concatenate([new_buf[start:], new_buf[:start]])

    mu = float(np.mean(valid))
    sigma = float(np.std(valid, ddof=1))

    if sigma < 1e-10:
        z = 0.0
    else:
        z = (x - mu) / sigma

    return z, new_buf, new_count
