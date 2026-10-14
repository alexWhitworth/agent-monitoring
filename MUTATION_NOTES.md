# Mutation Testing Notes (F-013)

Restricted mutation testing per the v0 spec: `mutmut` runs against the three
critical math modules only. All other modules are excluded by user directive
(recorded per-feature in `plans/v0_spec.json`, `mutation_testing` fields).

## How to run

```bash
make mutate                   # dev-workflow gate (Makefile): fresh run
# or by hand:
uv pip install -e ".[dev]"   # mutmut>=3 is a dev extra
rm -rf mutants               # clear cached verdicts (see caveat below)
mutmut run                   # ~1-2 min; prints a live progress bar
mutmut results               # list surviving mutants
mutmut show <mutant_name>    # diff of a specific mutant
```

Caveat: mutmut caches verdicts under `mutants/`. After editing tests or source,
`rm -rf mutants` before re-running so every mutant is re-tested against the
new suite.

## Configuration

`[tool.mutmut]` in `pyproject.toml`:

- `source_paths = ["src"]` — the whole package is copied to `mutants/` so
  imports resolve, but mutation is restricted by `only_mutate` to
  `joint.py`, `population.py`, `queue.py`.
- `pytest_add_cli_args = ["--no-cov"]` — the project's pytest `addopts`
  enable pytest-cov with a `fail_under = 85` gate. Per-mutant runs execute
  only the covering tests, which would fail that gate and mark every mutant
  "killed" spuriously. `--no-cov` disables coverage for mutation runs only;
  normal `pytest` runs keep the gate.

`tests/conftest.py` loads a dedicated hypothesis profile when running under
mutmut: mutmut drives several in-process pytest sessions, and each builds a
fresh `unittest.TestCase` instance, which trips hypothesis's
`differing_executors` health check (a false positive here — the check targets
genuinely different executors such as pytest vs. Django runners). Normal
pytest runs keep all health checks active.

## Results (final run)

| Module         | Mutants | Killed | Survived (justified below) |
| -------------- | ------- | ------ | --------------------------- |
| joint.py       | 131     | 126    | 5                           |
| population.py  | 257     | 250    | 7                           |
| queue.py       | 204     | 187    | 17                          |
| **Total**      | **592** | **563 (95.1%)** | **29 (all justified)** |

The first hardening pass killed 68 of the original 97 survivors with exact
boundary (n-1/n/n+1), dtype-contract, state-preservation, exact-selection,
and independent-numpy-oracle assertions; the remaining 29 are equivalent
mutants, justified below.

## Justified survivors (equivalent mutants)

### joint.py — 5

| Mutant(s) | Mutation | Rationale |
| --- | --- | --- |
| `x_update_joint__mutmut_12`, `__mutmut_14` | `flags.writeable = None` | NumPy coerces `None` to `False` on this flag; the assignment is identical to the original. Verified empirically. |
| `x_update_joint__mutmut_13`, `__mutmut_15` | `flags.writeable = True` | Redundant no-op: `MonitorState.__post_init__` unconditionally re-freezes `mu`/`M2` via `_freeze_arrays` (types.py), so the mutated line cannot change observable state. Verified by applying the mutant and running the read-only assertions. |
| `x_update_joint__mutmut_86` | `flagged = D2 >= threshold` | Measure-zero boundary: distinguishes the original only when `D2` exactly equals `chi2.ppf(1-alpha, A)`, which is not constructible in floating point. Existing tests pin `flagged == (D2 > threshold)` for every non-boundary value. |

### population.py — 7

| Mutant(s) | Mutation | Rationale |
| --- | --- | --- |
| `x_score_sessions__mutmut_22`, `__mutmut_24` | `isinstance(...) or True` / `float(None)` | The `isinstance` union-narrowing guard is defensive typing for `mypy --strict` (`d_squared` returns `float | ndarray`); the scalar arm is unreachable because `sessions.F` is `(S, A)` per DS-005, so `d_squared` always returns an `(S,)` array. |
| `x_absorb__mutmut_78`, `__mutmut_80` | `new_features` dtype `None` / dropped | `np.empty` defaults to float64 — semantically identical. (The dtype contract is still asserted by `test_absorb_result_dtypes`, which kills the equivalent mutants on `ticks`/`tick_flagged`/`absorbed_at`.) |
| `x_population_stats__mutmut_35` | `rowvar=None` | `np.cov` treats `None` as falsy ⇒ `rowvar=False` — identical. Verified empirically. |
| `x_population_stats__mutmut_36`, `__mutmut_39` | `ddof=None` / `ddof` dropped | `ddof=None` is `np.cov`'s default and resolves to the unbiased `N-1` — identical to the explicit `ddof=1`. Verified empirically. |

### queue.py — 17

| Mutant(s) | Mutation | Rationale |
| --- | --- | --- |
| `__mutmut_23`, `__mutmut_25` | `cand_d2` dtype `None` / dropped | `np.array` infers float64 from a list of Python floats — identical. |
| `__mutmut_41`, `__mutmut_43` | `cand_features` dtype `None` / dropped | `np.zeros` defaults to float64 — identical. |
| `__mutmut_46`, `__mutmut_48` | `cand_flagged` dtype `None` / dropped | Internal scratch array; float64 `0.0`/`1.0` has the same truthiness as bool in every downstream use (`if cand_flagged[i]`) — no observable difference. |
| `__mutmut_56`, `__mutmut_58` | `.get(sc.id, None)` / default dropped | `None` assigned into a bool array becomes `False` — identical to the original `False` default. |
| `__mutmut_96` | `flagged_budget = flagged - cascade` | The incoming cascade is positive only when the top_k budget exceeded all S candidates — i.e. every candidate is already allocated to top_k and the flagged pool is empty. The budget value is then unobservable. |
| `__mutmut_122`, `__mutmut_124`, `__mutmut_125`, `__mutmut_126` | `A > 0` guard variants | `MonitorConfig` validation guarantees `A >= 2` (API-001), so `A > 0` is always true and the `else` branch is dead. |
| `__mutmut_129` | `per_axis_each < 0` | When `per_axis_each == 0` the mutant continues the loop but `min(0, len(pool))` selects zero sessions — the same empty stratum. |
| `__mutmut_176`, `__mutmut_185` | `uniform_budget > 0` boundary | With budget 0, `rng.choice(remaining, size=0)` returns an empty array — the same empty selection as the guarded path. |
| `__mutmut_180` | `replace=None` | `Generator.choice` treats `None` as falsy ⇒ `replace=False` — identical. Verified empirically. |

## Maintenance notes

During hardening, a latent flake in `test_welford_vs_numpy` (F-005) was
exposed and fixed: its fixed `atol` sat below the floating-point cancellation
noise floor for ±1e6-scale inputs, so it failed nondeterministically on fresh
hypothesis streams. The tolerances are now scale-aware (relative to the
oracle magnitude) and were verified stable across 10 distinct hypothesis
seeds. A flaky suite would invalidate mutation verdicts, so this fix is a
prerequisite for the workflow above.

## Excluded modules — rationale

Mutation testing deliberately not run on (per user directive, mirrored from
each feature's `mutation_testing.rationale_if_disabled` in `plans/v0_spec.json`):

| Module | Rationale |
| --- | --- |
| `types.py` | Zero logic: frozen dataclass declarations + trivial `__post_init__` array freezing (F-001). |
| `config.py` | Validation rules + JSON codec; every rule has accept/reject unit tests and round-trip property tests (F-001). |
| `agg.py` | Trivial pure functions with oracle-exact hypothesis tests (F-002). |
| `normalizer.py` | Covered by exact buffer-recomputed-oracle property tests and reference parity (F-003). |
| `axis.py` | Covered by stepwise EWMA oracle tests and reference parity (F-004). |
| `advance.py` | Thin orchestration; covered by reference parity and the [T1] streaming-vs-batch-replay property (F-006). |
| `checkpoint.py` | I/O edge; mutation testing on filesystem round-trips is low-value and flaky (F-010). |
| `monitor.py` | Stateful facade that delegates to the pure core; contains no detection logic (F-011). |
| `__init__.py` | Pure re-export module (imports + `__all__` only); zero logic to mutate. |
