# ruff: noqa: N806, N803
"""Joint Mahalanobis/chi-square anomaly detector (Steps 5-6).

API-006: welford_update(n, mu, M2, S) -> (n, mu, M2)
API-007: d_squared(S, mu, sigma_inv) -> float | np.ndarray
API-008: update_joint(state, S_vector) -> (state, JointResult)

Welford rank-one mean + scatter update; ridge-regularized Sigma inversion;
D2 and chi-square p-value/threshold with min_warm gate.
"""

from __future__ import annotations

__all__ = ["d_squared", "update_joint", "welford_update"]

import numpy as np
from scipy.stats import chi2  # type: ignore[import-untyped]

from agent_monitor.types import JointResult, MonitorState


def welford_update(
    n: int, mu: np.ndarray, M2: np.ndarray, S: np.ndarray
) -> tuple[int, np.ndarray, np.ndarray]:
    """Rank-one Welford mean + scatter update (Step 5a).

    Args:
        n: Current Welford sample count.
        mu: (A,) current mean vector.
        M2: (A, A) current scatter matrix.
        S: (A,) new score vector S(t).

    Returns:
        Updated (n, mu, M2).
    """
    n_new = n + 1
    d_old = S - mu
    mu_new = mu + d_old / n_new
    # M2 = sum (x_i - mu_{i-1}) (x_i - mu_i)^T  (rank-one update)
    M2_new = M2 + np.outer(d_old, S - mu_new)
    return n_new, mu_new, M2_new


def d_squared(
    S: np.ndarray, mu: np.ndarray, sigma_inv: np.ndarray
) -> float | np.ndarray:
    """Squared Mahalanobis distance: (S - mu)^T Sigma_inv (S - mu).

    Args:
        S: (A,) single vector or (S, A) batch.
        mu: (A,) mean vector.
        sigma_inv: (A, A) inverse covariance matrix.

    Returns:
        Scalar or (S,) array of non-negative squared distances.
    """
    diff = S - mu  # (A,) or (S, A)
    if diff.ndim == 1:
        return float(diff @ sigma_inv @ diff)
    # Batch: (S, A) @ (A, A) -> (S, A); elementwise multiply; sum over axis=1
    return np.sum(diff @ sigma_inv * diff, axis=1)


def update_joint(
    state: MonitorState, S_vector: np.ndarray
) -> tuple[MonitorState, JointResult]:
    """Steps 5-6: Welford update, covariance inversion, D2, chi-square test.

    Args:
        state: Current MonitorState.
        S_vector: (A,) float64 axis score vector (0.0-fill for warming axes).

    Returns:
        state: MonitorState with updated n, mu, M2.
        result: JointResult for this tick.

    Raises:
        numpy.linalg.LinAlgError if ridge-regularized Sigma is singular.
    """
    cfg = state.config
    A = len(cfg.axis_names)

    # Step 5a: Welford update
    n_new, mu_new, M2_new = welford_update(state.n, state.mu, state.M2, S_vector)

    # Build updated state
    mu_new.flags.writeable = False
    M2_new.flags.writeable = False
    new_state = MonitorState(
        config=state.config,
        tick=state.tick,
        metric_buf=state.metric_buf,
        metric_count=state.metric_count,
        theta=state.theta,
        axis_buf=state.axis_buf,
        axis_count=state.axis_count,
        n=n_new,
        mu=mu_new,
        M2=M2_new,
        population=state.population,
    )

    threshold = float(chi2.ppf(1.0 - cfg.alpha, df=A))

    if n_new < cfg.min_warm:
        return new_state, JointResult(
            n=n_new,
            D2=None,
            threshold=threshold,
            p_value=None,
            flagged=False,
            note=f"warming up ({n_new}/{cfg.min_warm})",
        )

    # Step 5b: Regularized covariance inverse
    Sigma = M2_new / (n_new - 1) + cfg.ridge * np.eye(A)
    try:
        sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            f"Ridge-regularized covariance is singular (ridge={cfg.ridge}); "
            "check for degenerate or constant axes"
        ) from exc

    # Step 6: D2 and chi-square test
    D2 = float(d_squared(S_vector, mu_new, sigma_inv))
    p_value = float(chi2.sf(D2, df=A))
    flagged = D2 > threshold

    return new_state, JointResult(
        n=n_new,
        D2=D2,
        threshold=threshold,
        p_value=p_value,
        flagged=flagged,
        note="ok" if not flagged else "joint anomaly detected",
    )
