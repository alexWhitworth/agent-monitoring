"""MonitorConfig & QueueConfig validation and JSON codec.

API-001: validate_monitor_config  - invariant checking + k resolution.
API-002: config_to_json / config_from_json - canonical JSON round-trip.
"""

from __future__ import annotations

__all__ = [
    "VALID_AGG_NAMES",
    "config_from_json",
    "config_to_json",
    "validate_monitor_config",
    "validate_queue_config",
]

import json
from typing import Any

from scipy.stats import chi2  # type: ignore[import-untyped]

from agent_monitor.types import MonitorConfig, QueueConfig

VALID_AGG_NAMES: frozenset[str] = frozenset({"mean", "max", "l2"})
"""Known aggregation function names."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_monitor_config(config: MonitorConfig) -> MonitorConfig:
    """Validate MonitorConfig invariants and resolve k=None to chi2_A(0.99).

    Returns a new MonitorConfig with k resolved (if None was provided).
    Raises ValueError on any invariant violation.
    """
    axis_names = config.axis_names
    metric_names = config.metric_names
    num_axes = len(axis_names)

    # A >= 2
    if num_axes < 2:
        raise ValueError(f"At least 2 axes required; got {num_axes}")

    # metric_names must have one entry per axis, each with >= 1 metric
    if len(metric_names) != num_axes:
        raise ValueError(
            f"metric_names length ({len(metric_names)}) must equal "
            f"axis_names length ({num_axes})"
        )
    for i, mnames in enumerate(metric_names):
        if len(mnames) < 1:
            raise ValueError(
                f"metric_names[{i}] ({axis_names[i]}) must have >= 1 metric names"
            )

    # window >= 2
    if config.window < 2:
        raise ValueError(f"window must be >= 2; got {config.window}")

    # 0 < lam <= 1
    if not (0.0 < config.lam <= 1.0):
        raise ValueError(f"lam must be in (0, 1]; got {config.lam}")

    # min_warm >= 2
    if config.min_warm < 2:
        raise ValueError(f"min_warm must be >= 2; got {config.min_warm}")

    # population_window >= 1
    if config.population_window < 1:
        raise ValueError(f"population_window must be >= 1; got {config.population_window}")

    # winsor_k >= 1
    if config.winsor_k < 1:
        raise ValueError(f"winsor_k must be >= 1; got {config.winsor_k}")

    # winsor_pct: None accepted, otherwise 0 < p < 1
    if config.winsor_pct is not None and not (0.0 < config.winsor_pct < 1.0):
        raise ValueError(
            f"winsor_pct must be None or in (0, 1); got {config.winsor_pct}"
        )

    # agg must be known
    if config.agg not in VALID_AGG_NAMES:
        raise ValueError(
            f"Unknown aggregation '{config.agg}'; must be one of {sorted(VALID_AGG_NAMES)}"
        )

    # k must be > 0 when provided (not None)
    if config.k is not None and config.k <= 0:
        raise ValueError(f"k must be > 0 when provided; got {config.k}")

    # Resolve k
    k_resolved = config.k if config.k is not None else float(chi2.ppf(0.99, df=num_axes))

    # If nothing changed (k was already set), return the original config
    if k_resolved == config.k:
        return config

    return MonitorConfig(
        axis_names=config.axis_names,
        metric_names=config.metric_names,
        window=config.window,
        lam=config.lam,
        k=k_resolved,
        alpha=config.alpha,
        min_warm=config.min_warm,
        ridge=config.ridge,
        agg=config.agg,
        population_window=config.population_window,
        min_population=config.min_population,
        winsor_pct=config.winsor_pct,
        winsor_k=config.winsor_k,
    )


def validate_queue_config(qcfg: QueueConfig) -> QueueConfig:
    """Validate QueueConfig: all budgets must be >= 0.

    Returns the config unchanged on success.
    Raises ValueError on any invariant violation.
    """
    for field_name in ("top_k", "flagged", "per_axis", "uniform", "reserved"):
        val = getattr(qcfg, field_name)
        if val < 0:
            raise ValueError(f"{field_name} must be >= 0; got {val}")
    return qcfg


# ---------------------------------------------------------------------------
# JSON codec (API-002)
# ---------------------------------------------------------------------------


def config_to_json(config: MonitorConfig) -> str:
    """Canonical JSON encode of MonitorConfig for checkpoint storage."""
    d: dict[str, Any] = {
        "axis_names": list(config.axis_names),
        "metric_names": [list(m) for m in config.metric_names],
        "window": config.window,
        "lam": config.lam,
        "k": config.k,
        "alpha": config.alpha,
        "min_warm": config.min_warm,
        "ridge": config.ridge,
        "agg": config.agg,
        "population_window": config.population_window,
        "min_population": config.min_population,
        "winsor_pct": config.winsor_pct,
        "winsor_k": config.winsor_k,
    }
    return json.dumps(d, sort_keys=True)


def config_from_json(json_str: str) -> MonitorConfig:
    """Decode MonitorConfig from a config_to_json-format JSON string.

    Raises ValueError on malformed JSON or unknown agg name.
    """
    try:
        d = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed config JSON: {exc}") from exc

    if not isinstance(d, dict):
        raise ValueError("Config JSON must be a JSON object")

    # Validate required keys
    required = {
        "axis_names", "metric_names", "window", "lam", "k", "alpha",
        "min_warm", "ridge", "agg", "population_window", "min_population",
        "winsor_pct", "winsor_k",
    }
    missing = required - d.keys()
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")

    # Validate agg is known
    agg = d["agg"]
    if not isinstance(agg, str) or agg not in VALID_AGG_NAMES:
        raise ValueError(f"Unknown aggregation '{agg}'; must be one of {sorted(VALID_AGG_NAMES)}")

    config = MonitorConfig(
        axis_names=tuple(d["axis_names"]),
        metric_names=tuple(tuple(m) for m in d["metric_names"]),
        window=int(d["window"]),
        lam=float(d["lam"]),
        k=float(d["k"]) if d["k"] is not None else None,
        alpha=float(d["alpha"]),
        min_warm=int(d["min_warm"]),
        ridge=float(d["ridge"]),
        agg=str(agg),
        population_window=int(d["population_window"]),
        min_population=int(d["min_population"]),
        winsor_pct=float(d["winsor_pct"]) if d["winsor_pct"] is not None else None,
        winsor_k=int(d["winsor_k"]),
    )
    return validate_monitor_config(config)
