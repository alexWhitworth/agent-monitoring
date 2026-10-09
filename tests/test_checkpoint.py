# ruff: noqa: N806
"""Unit tests for checkpoint.py — save/load round-trip, format version gate,
atomic write, shape validation, and edge cases (F-010).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agent_monitor.checkpoint import (
    FORMAT_VERSION,
    load_checkpoint,
    save_checkpoint,
    _ndarray_from_bytes,
    _ndarray_to_bytes,
)
from agent_monitor.config import config_to_json, validate_monitor_config
from agent_monitor.population import absorb
from agent_monitor.types import (
    AxisResult,
    JointResult,
    MonitorConfig,
    MonitorState,
    SessionFeatures,
    SessionWindow,
    TickResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    axis_names: tuple[str, ...] = ("Safety", "Quality"),
    metric_names: tuple[tuple[str, ...], ...] = (("m1", "m2"), ("m1",)),
    window: int = 80,
    lam: float = 0.25,
    min_warm: int = 30,
    population_window: int = 80,
    min_population: int = 5,
    winsor_pct: float | None = 0.99,
    winsor_k: int = 3,
    **overrides: object,
) -> MonitorConfig:
    return validate_monitor_config(
        MonitorConfig(
            axis_names=axis_names,
            metric_names=metric_names,
            window=window,
            lam=lam,
            min_warm=min_warm,
            population_window=population_window,
            min_population=min_population,
            winsor_pct=winsor_pct,
            winsor_k=winsor_k,
            **overrides,
        )
    )


def _make_state(
    config: MonitorConfig | None = None,
    tick: int = 42,
    with_theta: bool = True,
    with_population: bool = False,
) -> MonitorState:
    if config is None:
        config = _make_config()
    A = len(config.axis_names)
    M_max = max(len(m) for m in config.metric_names)
    w = config.window

    metric_buf = np.random.default_rng(1).normal(0, 1, size=(A, M_max, w)).astype(np.float64)
    metric_count = np.full((A, M_max), w, dtype=np.int64)
    theta = np.random.default_rng(2).normal(0, 0.5, size=(A,)).astype(np.float64) if with_theta else None
    axis_buf = np.random.default_rng(3).normal(0, 1, size=(A, w)).astype(np.float64)
    axis_count = np.full((A,), w, dtype=np.int64)
    mu = np.random.default_rng(4).normal(0, 0.5, size=(A,)).astype(np.float64)
    M2 = np.eye(A, dtype=np.float64) * 0.5

    # Population
    if with_population:
        rng = np.random.default_rng(5)
        pop_n = 10
        pop = SessionWindow(
            ids=np.array([f"sess_{i}" for i in range(pop_n)], dtype=object),
            ticks=np.full(pop_n, tick - 5, dtype=np.int64),
            features=rng.normal(0, 1, size=(pop_n, A)).astype(np.float64),
            tick_flagged=np.zeros(pop_n, dtype=bool),
            absorbed_at=np.full(pop_n, tick - 5, dtype=np.int64),
        )
    else:
        pop = SessionWindow(
            ids=np.array([], dtype=object),
            ticks=np.array([], dtype=np.int64),
            features=np.empty((0, A), dtype=np.float64),
            tick_flagged=np.array([], dtype=bool),
            absorbed_at=np.array([], dtype=np.int64),
        )

    return MonitorState(
        config=config,
        tick=tick,
        metric_buf=metric_buf,
        metric_count=metric_count,
        theta=theta,
        axis_buf=axis_buf,
        axis_count=axis_count,
        n=100,
        mu=mu,
        M2=M2,
        population=pop,
    )


# ---------------------------------------------------------------------------
# Array serialization helpers
# ---------------------------------------------------------------------------


class TestArraySerialization:
    def test_ndarray_round_trip(self) -> None:
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        blob = _ndarray_to_bytes(arr)
        restored = _ndarray_from_bytes(blob)
        assert restored is not None
        np.testing.assert_array_equal(arr, restored)
        assert not restored.flags.writeable

    def test_none_round_trip(self) -> None:
        assert _ndarray_from_bytes(_ndarray_to_bytes(None)) is None
        assert _ndarray_from_bytes(b"") is None

    def test_int_array_round_trip(self) -> None:
        arr = np.array([1, 2, 3], dtype=np.int64)
        blob = _ndarray_to_bytes(arr)
        restored = _ndarray_from_bytes(blob)
        np.testing.assert_array_equal(arr, restored)

    def test_3d_array_round_trip(self) -> None:
        arr = np.random.default_rng(0).normal(0, 1, size=(3, 4, 80)).astype(np.float64)
        blob = _ndarray_to_bytes(arr)
        restored = _ndarray_from_bytes(blob)
        np.testing.assert_array_equal(arr, restored)


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    def test_round_trip_empty_population(self, tmp_path: Path) -> None:
        state = _make_state()
        checkpoint_dir = tmp_path / "checkpoint"
        save_checkpoint(state, checkpoint_dir)

        assert checkpoint_dir.is_dir()
        assert (checkpoint_dir / "state.parquet").is_file()
        assert (checkpoint_dir / "population.parquet").is_file()

        restored = load_checkpoint(checkpoint_dir)
        assert restored.config == state.config
        assert restored.tick == state.tick
        np.testing.assert_array_equal(restored.metric_buf, state.metric_buf)
        np.testing.assert_array_equal(restored.metric_count, state.metric_count)
        if state.theta is not None and restored.theta is not None:
            np.testing.assert_array_equal(restored.theta, state.theta)
        else:
            assert restored.theta is None and state.theta is None
        np.testing.assert_array_equal(restored.axis_buf, state.axis_buf)
        np.testing.assert_array_equal(restored.axis_count, state.axis_count)
        assert restored.n == state.n
        np.testing.assert_array_equal(restored.mu, state.mu)
        np.testing.assert_array_equal(restored.M2, state.M2)

    def test_round_trip_with_population(self, tmp_path: Path) -> None:
        state = _make_state(with_population=True)
        checkpoint_dir = tmp_path / "checkpoint"
        save_checkpoint(state, checkpoint_dir)
        restored = load_checkpoint(checkpoint_dir)

        # Population arrays
        np.testing.assert_array_equal(restored.population.ids, state.population.ids)
        np.testing.assert_array_equal(restored.population.ticks, state.population.ticks)
        np.testing.assert_array_equal(restored.population.features, state.population.features)
        np.testing.assert_array_equal(restored.population.tick_flagged, state.population.tick_flagged)
        np.testing.assert_array_equal(restored.population.absorbed_at, state.population.absorbed_at)

    def test_round_trip_theta_none(self, tmp_path: Path) -> None:
        state = _make_state(with_theta=False)
        checkpoint_dir = tmp_path / "checkpoint"
        save_checkpoint(state, checkpoint_dir)
        restored = load_checkpoint(checkpoint_dir)
        assert restored.theta is None

    def test_arrays_read_only_after_load(self, tmp_path: Path) -> None:
        state = _make_state(with_population=True)
        checkpoint_dir = tmp_path / "checkpoint"
        save_checkpoint(state, checkpoint_dir)
        restored = load_checkpoint(checkpoint_dir)

        for arr_name in ("metric_buf", "metric_count", "axis_buf", "axis_count", "mu", "M2"):
            arr = getattr(restored, arr_name)
            assert not arr.flags.writeable, f"{arr_name} should be read-only"
        if restored.theta is not None:
            assert not restored.theta.flags.writeable

        # Population arrays
        assert not restored.population.features.flags.writeable
        assert not restored.population.ticks.flags.writeable
        assert not restored.population.tick_flagged.flags.writeable
        assert not restored.population.absorbed_at.flags.writeable

    def test_round_trip_twice_idempotent(self, tmp_path: Path) -> None:
        state = _make_state(with_population=True)
        cp = tmp_path / "cp1"
        cp2 = tmp_path / "cp2"
        save_checkpoint(state, cp)
        restored = load_checkpoint(cp)
        save_checkpoint(restored, cp2)
        restored2 = load_checkpoint(cp2)
        np.testing.assert_array_equal(restored2.metric_buf, state.metric_buf)
        assert restored2.config == state.config


# ---------------------------------------------------------------------------
# Format version gate
# ---------------------------------------------------------------------------


class TestFormatVersion:
    def test_saved_checkpoint_has_version_1(self, tmp_path: Path) -> None:
        state = _make_state()
        checkpoint_dir = tmp_path / "checkpoint"
        save_checkpoint(state, checkpoint_dir)
        table = pq.read_table(str(checkpoint_dir / "state.parquet"))
        assert int(table.column("format_version")[0]) == FORMAT_VERSION

    def test_rejects_unknown_version(self, tmp_path: Path) -> None:
        state = _make_state()
        checkpoint_dir = tmp_path / "checkpoint"
        save_checkpoint(state, checkpoint_dir)

        # Corrupt the version
        table = pq.read_table(str(checkpoint_dir / "state.parquet"))
        col_dict = table.to_pydict()
        col_dict["format_version"][0] = 99
        new_table = pa.table(col_dict, schema=table.schema)
        pq.write_table(new_table, str(checkpoint_dir / "state.parquet"), compression="zstd")

        with pytest.raises(ValueError, match="Unknown checkpoint format_version"):
            load_checkpoint(checkpoint_dir)


# ---------------------------------------------------------------------------
# Atomic write: temp-dir + rename
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    def test_atomic_write_on_success(self, tmp_path: Path) -> None:
        state = _make_state()
        checkpoint_dir = tmp_path / "checkpoint"
        save_checkpoint(state, checkpoint_dir)
        # No temp dir remains
        temp_dirs = [d for d in tmp_path.iterdir() if d.name.startswith(".checkpoint_")]
        assert len(temp_dirs) == 0

    def test_cleanup_on_failure(self, tmp_path: Path) -> None:
        """Injected mid-write failure leaves pre-existing checkpoint intact."""
        state = _make_state()
        cp_original = tmp_path / "original"
        save_checkpoint(state, cp_original)

        # Now try to overwrite but inject failure
        cp_target = tmp_path / "target"

        # Create a state that will fail during write by writing an invalid path
        import os

        with pytest.raises(OSError):
            # Force failure by making the parent directory read-only
            os.chmod(str(tmp_path), 0o500)
            try:
                save_checkpoint(state, cp_target)
            finally:
                os.chmod(str(tmp_path), 0o700)

        # Original checkpoint intact
        assert cp_original.exists()
        restored = load_checkpoint(cp_original)
        assert restored.tick == state.tick


# ---------------------------------------------------------------------------
# Shape validation on load
# ---------------------------------------------------------------------------


class TestShapeValidation:
    def test_mismatched_config_rejected(self, tmp_path: Path) -> None:
        state = _make_state(
            config=_make_config(
                axis_names=("a", "b", "c"),
                metric_names=(("m",), ("m",), ("m",)),
            )
        )
        checkpoint_dir = tmp_path / "checkpoint"
        save_checkpoint(state, checkpoint_dir)

        # Modify config_json to have different axis count
        table = pq.read_table(str(checkpoint_dir / "state.parquet"))
        col_dict = table.to_pydict()
        wrong_config = _make_config(axis_names=("x", "y"))
        col_dict["config_json"][0] = config_to_json(wrong_config)
        new_table = pa.table(col_dict, schema=table.schema)
        pq.write_table(new_table, str(checkpoint_dir / "state.parquet"), compression="zstd")

        with pytest.raises(ValueError, match="shape mismatch"):
            load_checkpoint(checkpoint_dir)

    def test_missing_files_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nonexistent")

    def test_missing_state_parquet_raises(self, tmp_path: Path) -> None:
        state = _make_state()
        checkpoint_dir = tmp_path / "checkpoint"
        save_checkpoint(state, checkpoint_dir)
        (checkpoint_dir / "state.parquet").unlink()
        with pytest.raises(FileNotFoundError, match="state.parquet"):
            load_checkpoint(checkpoint_dir)

    def test_missing_population_parquet_raises(self, tmp_path: Path) -> None:
        state = _make_state()
        checkpoint_dir = tmp_path / "checkpoint"
        save_checkpoint(state, checkpoint_dir)
        (checkpoint_dir / "population.parquet").unlink()
        with pytest.raises(FileNotFoundError, match="population.parquet"):
            load_checkpoint(checkpoint_dir)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_large_population_window(self, tmp_path: Path) -> None:
        """Checkpoint with large population (~1000 rows)."""
        config = _make_config(
            axis_names=("a", "b", "c"),
            metric_names=(("m",), ("m",), ("m",)),
        )
        state = _make_state(config=config, with_population=True)
        # Add more population rows
        A = 3
        rng = np.random.default_rng(99)
        n_extra = 990
        sf = SessionFeatures(
            ids=tuple(f"large_{i}" for i in range(n_extra)),
            tick=state.tick,
            F=rng.normal(0, 1, size=(n_extra, A)).astype(np.float64),
        )
        state = absorb(state, sf, tick_flagged=False)

        cp = tmp_path / "checkpoint"
        save_checkpoint(state, cp)
        restored = load_checkpoint(cp)
        assert len(restored.population.ids) == len(state.population.ids)

    def test_fresh_state_no_population(self, tmp_path: Path) -> None:
        state = _make_state(with_population=False)
        cp = tmp_path / "checkpoint"
        save_checkpoint(state, cp)
        restored = load_checkpoint(cp)
        assert len(restored.population.ids) == 0
        assert restored.tick == state.tick

    def test_overwrite_existing_checkpoint(self, tmp_path: Path) -> None:
        state1 = _make_state(tick=10)
        state2 = _make_state(tick=20)
        cp = tmp_path / "checkpoint"
        save_checkpoint(state1, cp)
        save_checkpoint(state2, cp)
        restored = load_checkpoint(cp)
        assert restored.tick == 20

    def test_corrupted_population_features_raises(self, tmp_path: Path) -> None:
        """Corrupted features_binary blob should raise on load."""
        state = _make_state(with_population=True)
        cp = tmp_path / "checkpoint"
        save_checkpoint(state, cp)

        table = pq.read_table(str(cp / "population.parquet"))
        col_dict = table.to_pydict()
        # Corrupt the first row's features_binary
        col_dict["features_binary"][0] = b"corrupted_data_not_valid_npy"
        new_table = pa.table(col_dict, schema=table.schema)
        pq.write_table(new_table, str(cp / "population.parquet"), compression="zstd")

        with pytest.raises(ValueError):
            load_checkpoint(cp)

    def test_special_config_values(self, tmp_path: Path) -> None:
        """Config with k=None, non-default lambda, etc."""
        config = _make_config(k=None, lam=0.5, agg="max", winsor_pct=None)
        state = _make_state(config=config)
        cp = tmp_path / "checkpoint"
        save_checkpoint(state, cp)
        restored = load_checkpoint(cp)
        assert restored.config == config
        assert restored.config.k is not None  # k=None resolved to chi2.ppf
        assert restored.config.agg == "max"
        assert restored.config.winsor_pct is None
        assert restored.config.lam == 0.5