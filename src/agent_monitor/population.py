# ruff: noqa: N806
"""Population window with dedup, eviction, winsorization, and stats.

API-010: absorb(state, sessions, tick_flagged) -> state
API-011: winsor_caps(state) -> caps | None
API-012: population_stats(state) -> (mu, Sigma_inv, n) | None
API-013: score_sessions(state, sessions) -> tuple[SessionScore, ...]
"""

from __future__ import annotations

__all__ = [
    "absorb",
    "population_stats",
    "score_sessions",
    "winsor_caps",
]

import numpy as np

from agent_monitor.joint import d_squared
from agent_monitor.types import MonitorState, SessionFeatures, SessionScore, SessionWindow


def absorb(
    state: MonitorState, sessions: SessionFeatures, tick_flagged: bool
) -> MonitorState:
    """Append session rows, dedup on session_id, evict by retention cutoff.

    retention = max(population_window, window * winsor_k).
    Rows with tick < state.tick - retention + 1 are evicted.
    """
    cfg = state.config
    retention = max(cfg.population_window, cfg.window * cfg.winsor_k)
    cutoff = state.tick - retention + 1

    pop = state.population
    existing_ids = set(pop.ids)

    # Filter incoming: dedup within batch and vs retained
    new_rows: list[tuple[str, int, np.ndarray, bool, int]] = []
    batch_ids: set[str] = set()
    for i, sid in enumerate(sessions.ids):
        if sid in existing_ids or sid in batch_ids:
            continue
        batch_ids.add(sid)
        new_rows.append(
            (sid, sessions.tick, sessions.F[i].copy(), tick_flagged, state.tick)
        )

    if not new_rows:
        # Just evict
        keep_mask = pop.ticks >= cutoff
        if not np.all(keep_mask):
            new_pop = SessionWindow(
                ids=pop.ids[keep_mask],
                ticks=pop.ticks[keep_mask],
                features=pop.features[keep_mask],
                tick_flagged=pop.tick_flagged[keep_mask],
                absorbed_at=pop.absorbed_at[keep_mask],
            )
            return MonitorState(
                config=state.config,
                tick=state.tick,
                metric_buf=state.metric_buf,
                metric_count=state.metric_count,
                theta=state.theta,
                axis_buf=state.axis_buf,
                axis_count=state.axis_count,
                n=state.n,
                mu=state.mu,
                M2=state.M2,
                population=new_pop,
            )
        return state

    # Concatenate new rows
    n_existing = len(pop.ids)
    n_new = len(new_rows)
    new_ids = np.empty(n_existing + n_new, dtype=object)
    new_ticks = np.empty(n_existing + n_new, dtype=np.int64)
    new_features = np.empty((n_existing + n_new, len(cfg.axis_names)), dtype=np.float64)
    new_tick_flagged = np.empty(n_existing + n_new, dtype=bool)
    new_absorbed_at = np.empty(n_existing + n_new, dtype=np.int64)

    new_ids[:n_existing] = pop.ids
    new_ticks[:n_existing] = pop.ticks
    new_features[:n_existing] = pop.features
    new_tick_flagged[:n_existing] = pop.tick_flagged
    new_absorbed_at[:n_existing] = pop.absorbed_at

    for j, (sid, t, f, tf, aa) in enumerate(new_rows):
        new_ids[n_existing + j] = sid
        new_ticks[n_existing + j] = t
        new_features[n_existing + j] = f
        new_tick_flagged[n_existing + j] = tf
        new_absorbed_at[n_existing + j] = aa

    # Evict
    keep_mask = new_ticks >= cutoff
    new_pop = SessionWindow(
        ids=new_ids[keep_mask],
        ticks=new_ticks[keep_mask],
        features=new_features[keep_mask],
        tick_flagged=new_tick_flagged[keep_mask],
        absorbed_at=new_absorbed_at[keep_mask],
    )

    return MonitorState(
        config=state.config,
        tick=state.tick,
        metric_buf=state.metric_buf,
        metric_count=state.metric_count,
        theta=state.theta,
        axis_buf=state.axis_buf,
        axis_count=state.axis_count,
        n=state.n,
        mu=state.mu,
        M2=state.M2,
        population=new_pop,
    )


def winsor_caps(state: MonitorState) -> np.ndarray | None:
    """Per-axis two-sided clip bounds [q_{1-p}, q_p] over full retention window.

    None while winsor_pct is None or retained rows < 100.
    Returns (2, A) array: caps[0] = lower, caps[1] = upper.
    """
    cfg = state.config
    if cfg.winsor_pct is None:
        return None

    pop = state.population
    if len(pop.ids) < 100:
        return None

    p = cfg.winsor_pct
    lower = np.quantile(pop.features, 1.0 - p, axis=0)
    upper = np.quantile(pop.features, p, axis=0)
    caps = np.stack([lower, upper])
    return caps


def population_stats(state: MonitorState) -> tuple[np.ndarray, np.ndarray, int] | None:
    """(mu, Sigma_inv, n) over last population_window ticks of winsorized rows.

    Rows clipped to winsor_caps; raw rows while caps is None.
    None until n >= min_population.
    """
    cfg = state.config
    pop = state.population

    current_tick = state.tick
    cov_cutoff = current_tick - cfg.population_window + 1
    cov_mask = pop.ticks >= cov_cutoff
    rows = pop.features[cov_mask]

    # Winsorize if caps are available
    caps = winsor_caps(state)
    if caps is not None:
        clipped = np.clip(rows, caps[0], caps[1])
    else:
        clipped = rows

    n = clipped.shape[0]
    if n < cfg.min_population:
        return None

    mu = np.mean(clipped, axis=0)
    cov = np.cov(clipped, rowvar=False, ddof=1)
    A = len(cfg.axis_names)
    Sigma = cov + cfg.ridge * np.eye(A)
    try:
        sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            f"Population ridge-regularized covariance is singular (ridge={cfg.ridge})"
        ) from exc

    return mu, sigma_inv, n


def score_sessions(
    state: MonitorState, sessions: SessionFeatures
) -> tuple[SessionScore, ...]:
    """Batch d2 + chi-square p-values for candidate sessions vs population.

    Candidates scored RAW (never clipped). d2/p_value None while
    population n < min_population.
    """
    from scipy.stats import chi2  # type: ignore[import-untyped]

    stats = population_stats(state)
    if stats is None:
        return tuple(
            SessionScore(id=sid, d2=None, p_value=None) for sid in sessions.ids
        )

    mu, sigma_inv, _n_pop = stats
    A = len(state.config.axis_names)
    d2_vals = d_squared(sessions.F, mu, sigma_inv)  # (S,) array

    scores: list[SessionScore] = []
    for i, sid in enumerate(sessions.ids):
        d2 = float(d2_vals[i]) if isinstance(d2_vals, np.ndarray) else float(d2_vals)
        p_value = float(chi2.sf(d2, df=A))
        scores.append(SessionScore(id=sid, d2=d2, p_value=p_value))

    return tuple(scores)
