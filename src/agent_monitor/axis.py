"""Per-axis EWMA anomaly detector (Steps 2-4).

API-005: update_axis(state, a, X_a, agg_fn) -> (state, AxisResult)

Step 2: S_A = agg_fn(valid z-scores)
Step 3: theta_A(t) = lam * S_A + (1-lam) * theta_A(t-1), cold-start at first S_A
Step 4: flag iff sigma_S > 1e-10 and |S_A - theta_A| > k * sigma_S
"""

from __future__ import annotations

__all__ = ["update_axis"]

from collections.abc import Callable, Sequence

import numpy as np

from agent_monitor.normalizer import zscore
from agent_monitor.types import AxisResult, MonitorState


def _rolling_mean_std(buf: np.ndarray, count: int) -> tuple[float, float]:
    """Compute mean and std (ddof=1) of valid entries in a ring buffer."""
    w = len(buf)
    if count <= w:
        valid = buf[:count]
    else:
        pos = (count - 1) % w
        start = (pos + 1) % w
        valid = np.concatenate([buf[start:], buf[:start]]) if start != 0 else buf
    mu = float(np.mean(valid))
    sigma = float(np.std(valid, ddof=1))
    return mu, sigma


def update_axis(
    state: MonitorState,
    a: int,
    X_a: Sequence[float],  # noqa: N803
    agg_fn: Callable[[Sequence[float]], float],
) -> tuple[MonitorState, AxisResult]:
    """Steps 2-4 for one axis: z-score metrics, aggregate, EWMA, flag.

    Args:
        state: Current MonitorState.
        a: Axis index (0 <= a < A).
        X_a: M_a raw metric readings this tick (finite float64-compatible).
        agg_fn: Aggregation function from agg.get_agg().

    Returns:
        state: MonitorState with updated axis-level buffers and theta for axis a.
        result: AxisResult for this tick.

    Raises:
        ValueError if X_a contains non-finite values.
    """
    cfg = state.config
    axis_name = cfg.axis_names[a]
    M_a = len(cfg.metric_names[a])  # noqa: N806

    if len(X_a) != M_a:
        raise ValueError(
            f"Axis '{axis_name}' expects {M_a} metrics, got {len(X_a)}"
        )

    # Step 1: z-score each metric via normalizer
    zs: list[float] = []
    metric_buf = np.copy(state.metric_buf[a])  # (M_max, w)
    metric_count = np.copy(state.metric_count[a])  # (M_max,)

    for m_idx, x in enumerate(X_a):
        x_f = float(x)
        if not np.isfinite(x_f):
            raise ValueError(
                f"Axis '{axis_name}' metric {m_idx}: non-finite value {x_f}"
            )
        z, new_buf_m, new_count_m = zscore(
            metric_buf[m_idx], int(metric_count[m_idx]), x_f
        )
        metric_buf[m_idx] = new_buf_m
        metric_count[m_idx] = new_count_m
        if z is not None:
            zs.append(z)

    # Step 2: S_A = agg_fn(valid z-scores); None if no valid z-scores
    S_A: float | None = None  # noqa: N806
    if zs:
        S_A = agg_fn(zs)  # noqa: N806

    # Update state's per-axis buffers
    axis_buf = np.copy(state.axis_buf[a])
    axis_count = int(state.axis_count[a])
    theta_val = state.theta[a] if state.theta is not None else None

    note = "ok"
    flagged = False
    sigma_S: float | None = None  # noqa: N806
    deviation: float | None = None
    threshold: float | None = None
    theta_out: float | None = theta_val

    if S_A is None:
        note = "warming up z-scores"
    else:
        # Step 3: EWMA theta_A
        if theta_val is None:
            theta_out = S_A  # cold start
        else:
            theta_out = cfg.lam * S_A + (1.0 - cfg.lam) * theta_val

        # Write S_A into axis ring buffer
        pos = axis_count % cfg.window
        axis_buf[pos] = S_A
        axis_count += 1

        # Step 4: adaptive threshold test
        if axis_count >= 2:
            _mu_S, sigma_S = _rolling_mean_std(axis_buf, axis_count)  # noqa: N806
            if sigma_S > 1e-10:
                deviation = abs(S_A - theta_out)
                threshold = cfg.k * sigma_S  # type: ignore[operator] # k is resolved
                if deviation > threshold:
                    flagged = True
                    if note == "ok":
                        note = "per-axis anomaly detected"
            elif note == "ok":
                note = "sigma_S near zero; threshold suppressed"
        else:
            note = "warming up sigma_S"

    # Build updated state: new axis-level arrays
    new_metric_buf = np.copy(state.metric_buf)
    new_metric_buf[a] = metric_buf
    new_metric_buf.flags.writeable = False

    new_metric_count = np.copy(state.metric_count)
    new_metric_count[a] = metric_count
    new_metric_count.flags.writeable = False

    new_axis_buf = np.copy(state.axis_buf)
    new_axis_buf[a] = axis_buf
    new_axis_buf.flags.writeable = False

    new_axis_count = np.copy(state.axis_count)
    new_axis_count[a] = axis_count
    new_axis_count.flags.writeable = False

    new_theta: np.ndarray | None = None
    if state.theta is not None or theta_out is not None:
        new_theta = (
            np.copy(state.theta) if state.theta is not None else np.zeros(len(cfg.axis_names))
        )
        new_theta[a] = theta_out if theta_out is not None else 0.0
        new_theta.flags.writeable = False

    new_state = MonitorState(
        config=state.config,
        tick=state.tick,
        metric_buf=new_metric_buf,
        metric_count=new_metric_count,
        theta=new_theta,
        axis_buf=new_axis_buf,
        axis_count=new_axis_count,
        n=state.n,
        mu=state.mu,
        M2=state.M2,
        population=state.population,
    )

    result = AxisResult(
        axis=axis_name,
        S_A=S_A,
        theta=theta_out,
        sigma_S=sigma_S,
        deviation=deviation,
        threshold=threshold,
        flagged=flagged,
        note=note,
    )

    return new_state, result
