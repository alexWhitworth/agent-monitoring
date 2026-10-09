# ruff: noqa: N806
"""Parquet checkpoint persistence (F-010, APIs 015-016).

Atomic temp-dir + rename save; data-only load (no pickle, no code execution).
Format version 1. Rejects unknown format versions.
"""

from __future__ import annotations

__all__ = ["load_checkpoint", "save_checkpoint"]

import io
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from agent_monitor.config import config_from_json, config_to_json
from agent_monitor.types import MonitorConfig, MonitorState, SessionWindow


# ---------------------------------------------------------------------------
# Schema & serialization helpers
# ---------------------------------------------------------------------------

FORMAT_VERSION: int = 1

# State columns for the nested single-row state.parquet
STATE_COLUMNS: list[tuple[str, pa.DataType]] = [
    ("format_version", pa.int64()),
    ("config_json", pa.utf8()),
    ("tick", pa.int64()),
    ("metric_buf_binary", pa.binary()),
    ("metric_count_binary", pa.binary()),
    ("theta_binary", pa.binary()),
    ("axis_buf_binary", pa.binary()),
    ("axis_count_binary", pa.binary()),
    ("n", pa.int64()),
    ("mu_binary", pa.binary()),
    ("M2_binary", pa.binary()),
]

# Population columns for the tabular population.parquet
POPULATION_COLUMNS: list[tuple[str, pa.DataType]] = [
    ("session_id", pa.utf8()),
    ("tick", pa.int64()),
    ("features_binary", pa.binary()),
    ("tick_flagged", pa.bool_()),
    ("absorbed_at", pa.int64()),
]


def _ndarray_to_bytes(arr: np.ndarray | None) -> bytes:
    """Serialize a numpy array to bytes via .npy format (no pickle)."""
    if arr is None:
        return b""
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _ndarray_from_bytes(data: bytes | None) -> np.ndarray | None:
    """Deserialize a numpy array from .npy bytes."""
    if data is None or len(data) == 0:
        return None
    buf = io.BytesIO(data)
    arr: np.ndarray = np.load(buf, allow_pickle=False)
    arr.flags.writeable = False
    return arr


# ---------------------------------------------------------------------------
# save_checkpoint
# ---------------------------------------------------------------------------


def save_checkpoint(state: MonitorState, path: Path) -> None:
    """Serialize state to a Parquet checkpoint directory via temp dir + atomic rename.

    Args:
        state: Current MonitorState to persist.
        path: Target checkpoint directory path.

    Raises:
        OSError: On write failure.
    """
    target = Path(path)

    # Write to temp dir first, then atomically rename
    tmp_dir = Path(tempfile.mkdtemp(prefix=".checkpoint_", dir=target.parent))

    try:
        _write_state_parquet(state, tmp_dir / "state.parquet")
        _write_population_parquet(state.population, tmp_dir / "population.parquet")
        # Atomic rename
        if target.exists():
            # Remove existing checkpoint first
            shutil.rmtree(target)
        os.rename(tmp_dir, target)
    except Exception:  # pragma: no cover
        # Clean up temp dir on failure; existing checkpoint untouched
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise


def _write_state_parquet(state: MonitorState, filepath: Path) -> None:
    """Write the single-row nested state.parquet."""
    cfg_json = config_to_json(state.config)

    row: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "config_json": cfg_json,
        "tick": state.tick,
        "metric_buf_binary": _ndarray_to_bytes(state.metric_buf),
        "metric_count_binary": _ndarray_to_bytes(state.metric_count),
        "theta_binary": _ndarray_to_bytes(state.theta),
        "axis_buf_binary": _ndarray_to_bytes(state.axis_buf),
        "axis_count_binary": _ndarray_to_bytes(state.axis_count),
        "n": state.n,
        "mu_binary": _ndarray_to_bytes(state.mu),
        "M2_binary": _ndarray_to_bytes(state.M2),
    }

    schema = pa.schema(STATE_COLUMNS)
    table = pa.table({col: pa.array([row[col]], type=dt) for col, dt in STATE_COLUMNS}, schema=schema)
    pq.write_table(table, str(filepath), compression="zstd")


def _write_population_parquet(pop: SessionWindow, filepath: Path) -> None:
    """Write the tabular population.parquet."""
    n = len(pop.ids)
    if n == 0:
        schema = pa.schema(POPULATION_COLUMNS)
        table = pa.table(
            {
                "session_id": pa.array([], type=pa.utf8()),
                "tick": pa.array([], type=pa.int64()),
                "features_binary": pa.array([], type=pa.binary()),
                "tick_flagged": pa.array([], type=pa.bool_()),
                "absorbed_at": pa.array([], type=pa.int64()),
            },
            schema=schema,
        )
    else:
        features_blobs = [
            _ndarray_to_bytes(pop.features[i]) for i in range(n)
        ]
        table = pa.table(
            {
                "session_id": pa.array([str(x) for x in pop.ids], type=pa.utf8()),
                "tick": pa.array(pop.ticks, type=pa.int64()),
                "features_binary": pa.array(features_blobs, type=pa.binary()),
                "tick_flagged": pa.array(pop.tick_flagged, type=pa.bool_()),
                "absorbed_at": pa.array(pop.absorbed_at, type=pa.int64()),
            },
            schema=pa.schema(POPULATION_COLUMNS),
        )
    pq.write_table(table, str(filepath), compression="zstd")


# ---------------------------------------------------------------------------
# load_checkpoint
# ---------------------------------------------------------------------------


def load_checkpoint(path: Path) -> MonitorState:
    """Load + validate checkpoint; returns MonitorState with all arrays read-only.

    Args:
        path: Checkpoint directory containing state.parquet and population.parquet.

    Returns:
        MonitorState reconstructed from the checkpoint.

    Raises:
        FileNotFoundError: If the checkpoint directory or its files are missing.
        ValueError: On unknown format_version or shape/config mismatch.
        OSError: On corrupt/unreadable files.
    """
    target = Path(path)
    if not target.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {target}")

    state_file = target / "state.parquet"
    pop_file = target / "population.parquet"

    if not state_file.is_file():
        raise FileNotFoundError(f"state.parquet not found in checkpoint: {target}")
    if not pop_file.is_file():
        raise FileNotFoundError(f"population.parquet not found in checkpoint: {target}")

    state_table = pq.read_table(str(state_file))
    if len(state_table) != 1:
        raise ValueError(f"state.parquet must have exactly 1 row, got {len(state_table)}")  # pragma: no cover

    row = state_table.to_pydict()

    format_version = int(row["format_version"][0])
    if format_version != FORMAT_VERSION:
        raise ValueError(
            f"Unknown checkpoint format_version: {format_version}. "
            f"This library supports version {FORMAT_VERSION}."
        )

    config = config_from_json(str(row["config_json"][0]))
    tick = int(row["tick"][0])
    n_joint = int(row["n"][0])

    metric_buf = _ndarray_from_bytes(row["metric_buf_binary"][0])
    metric_count = _ndarray_from_bytes(row["metric_count_binary"][0])
    theta = _ndarray_from_bytes(row["theta_binary"][0] if row["theta_binary"][0] else None)
    axis_buf = _ndarray_from_bytes(row["axis_buf_binary"][0])
    axis_count = _ndarray_from_bytes(row["axis_count_binary"][0])
    mu = _ndarray_from_bytes(row["mu_binary"][0])
    M2 = _ndarray_from_bytes(row["M2_binary"][0])

    # Validate array shapes against config
    _validate_state_arrays(config, tick, metric_buf, metric_count, theta, axis_buf, axis_count, n_joint, mu, M2)
    # Type narrowing: _validate_state_arrays ensures non-None for required fields
    assert metric_buf is not None
    assert metric_count is not None
    assert axis_buf is not None
    assert axis_count is not None
    assert mu is not None
    assert M2 is not None

    # Load population
    population = _load_population_parquet(pop_file, config)

    return MonitorState(
        config=config,
        tick=tick,
        metric_buf=metric_buf,
        metric_count=metric_count,
        theta=theta,
        axis_buf=axis_buf,
        axis_count=axis_count,
        n=n_joint,
        mu=mu,
        M2=M2,
        population=population,
    )


def _validate_state_arrays(
    config: MonitorConfig,
    tick: int,
    metric_buf: np.ndarray | None,
    metric_count: np.ndarray | None,
    theta: np.ndarray | None,
    axis_buf: np.ndarray | None,
    axis_count: np.ndarray | None,
    n_joint: int,
    mu: np.ndarray | None,
    M2: np.ndarray | None,
) -> None:
    """Validate that loaded arrays match the config topology."""
    A = len(config.axis_names)
    M_max = max(len(m) for m in config.metric_names)
    w = config.window

    if metric_buf is None or metric_buf.shape != (A, M_max, w):
        raise ValueError(  # pragma: no cover
            f"metric_buf shape mismatch: expected {(A, M_max, w)}, got "
            f"{metric_buf.shape if metric_buf is not None else None}"
        )
    if metric_count is None or metric_count.shape != (A, M_max):
        raise ValueError(  # pragma: no cover
            f"metric_count shape mismatch: expected {(A, M_max)}, got "
            f"{metric_count.shape if metric_count is not None else None}"
        )
    if theta is not None and theta.shape != (A,):
        raise ValueError(f"theta shape mismatch: expected {(A,)}, got {theta.shape}")  # pragma: no cover
    if axis_buf is None or axis_buf.shape != (A, w):
        raise ValueError(  # pragma: no cover
            f"axis_buf shape mismatch: expected {(A, w)}, got "
            f"{axis_buf.shape if axis_buf is not None else None}"
        )
    if axis_count is None or axis_count.shape != (A,):
        raise ValueError(  # pragma: no cover
            f"axis_count shape mismatch: expected {(A,)}, got "
            f"{axis_count.shape if axis_count is not None else None}"
        )
    if mu is None or mu.shape != (A,):
        raise ValueError(f"mu shape mismatch: expected {(A,)}, got {mu.shape if mu is not None else None}")  # pragma: no cover
    if M2 is None or M2.shape != (A, A):
        raise ValueError(f"M2 shape mismatch: expected {(A, A)}, got {M2.shape if M2 is not None else None}")  # pragma: no cover


def _load_population_parquet(filepath: Path, config: MonitorConfig) -> SessionWindow:
    """Load and reconstruct SessionWindow from population.parquet."""
    table = pq.read_table(str(filepath))
    col_dict = table.to_pydict()

    n_rows = len(table)
    if n_rows == 0:
        A = len(config.axis_names)
        return SessionWindow(
            ids=np.array([], dtype=object),
            ticks=np.array([], dtype=np.int64),
            features=np.empty((0, A), dtype=np.float64),
            tick_flagged=np.array([], dtype=bool),
            absorbed_at=np.array([], dtype=np.int64),
        )

    A = len(config.axis_names)
    ids_list = [str(x) for x in col_dict["session_id"]]
    ids = np.array(ids_list, dtype=object)

    ticks_arr = np.array(col_dict["tick"], dtype=np.int64)
    tick_flagged_arr = np.array(col_dict["tick_flagged"], dtype=bool)
    absorbed_at_arr = np.array(col_dict["absorbed_at"], dtype=np.int64)

    features_list: list[np.ndarray] = []
    for i in range(n_rows):
        blob = col_dict["features_binary"][i]
        feat = _ndarray_from_bytes(blob)
        if feat is None:
            raise ValueError(f"Corrupted features_binary for row {i}")  # pragma: no cover
        if feat.shape != (A,):
            raise ValueError(  # pragma: no cover
                f"Population row {i} features shape mismatch: expected {(A,)}, got {feat.shape}"
            )
        features_list.append(feat)

    features = np.stack(features_list).astype(np.float64)

    return SessionWindow(
        ids=ids,
        ticks=ticks_arr,
        features=features,
        tick_flagged=tick_flagged_arr,
        absorbed_at=absorbed_at_arr,
    )