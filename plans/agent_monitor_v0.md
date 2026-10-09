# Project: agent_monitor

## 1. System Overview

`agent_monitor` is a Python library implementing the **Adaptive Multi-Dimensional
Monitoring (AMDM)** algorithm (Shukla, 2025) for agentic AI systems, with a
two-tier architecture for production monitoring and human-review prioritization.

**Deployment context:** production streaming services and scheduled batch jobs
(e.g., Airflow `PythonOperator`). Data loading is **out of scope** — callers
load raw logs / LLM-judge outputs from S3 or an RDBMS themselves and hand the
library typed metric matrices.

```
                        ┌─────────────────────────────────────────────┐
 caller (S3 / RDBMS /   │  per tick t:  X ∈ ℝ^{A×M}  (complete,      │
 Airflow) prepares →    │               pre-aligned metric matrix)     │
                        └─────────────────────────────────────────────┘
                                          │
                ┌─────────────────────────▼─────────────────────────┐
                │  TIER 1 — AMDM aggregate monitor (streaming)      │
                │  1. rolling z-scores per metric                   │
                │  2. aggregate z's → axis score S_A (pluggable)   │
                │  3. EWMA baseline θ_A                            │
                │  4. flag axis if |S_A − θ_A| > k·σ_{S_A}         │
                │  5. Welford update of joint μ, Σ over S(t)       │
                │  6. flag joint if D² > χ²_A(1−α)                 │
                └─────────────────────────┬─────────────────────────┘
                                          │
        ┌─────────────────────────────────┴──────────────────────────┐
        ▼                                                              ▼
┌──────────────────────────┐               ┌────────────────────────────────────┐
│ TIER 2 — session scoring │               │  Review queue stratification      │
│ d²_s = (x_s−μ̂)ᵀ Σ̂⁻¹ x_s−μ̂│   ───────►    │  top-k d² | flagged-bucket |      │
│ vs rolling population of │               │  per-axis worst | uniform random │
│ past session vectors      │               │  (+ analyst reserve)              │
└──────────────────────────┘               └────────────────────────────────────┘
        │
        ▼
   immutable MonitorState ──checkpoint()──► Parquet directory (S3 / local)
```

**Architectural style — Hybrid FP/OOP:**
- All computation is expressed as **pure functions**: `(state, input) → (state', result)`,
  where `state` is a frozen-dataclass *value* (NumPy arrays marked read-only).
- Side effects (Parquet I/O) live strictly at the edges.
- A thin stateful `Monitor` facade exists purely for convenience in long-lived
  streaming services; it threads the immutable snapshot and adds no logic.
- **No Rust.** All hot paths are vectorized NumPy (BLAS-backed); PyO3 is an
  escape hatch behind stable array-in/array-out contracts if v1 calibration
  replay ever demands it.

## 2. Tech Stack & Dependencies

| Concern | Choice |
|---|---|
| Language / runtime | Python 3.13, `uv`-managed |
| Numeric core | `numpy` |
| χ² quantiles / p-values | `scipy.stats.chi2` |
| Checkpoint persistence | `pyarrow` → **Parquet** (directory: `state.parquet` + `population.parquet`) |
| Packaging | `src/` layout, `pyproject.toml`, hatchling |
| Lint / types | `ruff` (line 100, py313), `mypy --strict` |
| Tests | `pytest`, `pytest-cov` (fail_under = 85), `hypothesis` (critical math only) |
| Mutation testing | `mutmut`, **restricted to critical modules only**: `joint.py`, `population.py`, `queue.py` |

Deferred to later versions: calibration utilities (v1, needs real data),
Rust kernels (only if profiling justifies), plotnine ad-hoc reporting (v2),
partial/late-arriving axes handling (v1+, see §7).

## 3. Data Schema / Type Definitions

All models are `@dataclass(frozen=True)`. Pseudocode; exact field details below.

### 3.1 Configuration

```python
@dataclass(frozen=True)
class MonitorConfig:
    axis_names: tuple[str, ...]        # A axes, e.g. ("safety","quality",...)
    metric_names: tuple[tuple[str, ...], ...]  # M_a metric names per axis
    window: int = 80                   # w
    lam: float = 0.25                   # λ EWMA smoothing
    k: float | None = None              # per-axis multiplier; default χ²_A(0.99)
    alpha: float = 0.01                 # joint false-alarm rate
    min_warm: int = 30                 # joint detector warm-up ticks
    ridge: float = 1e-6                # covariance regularization
    agg: str = "mean"                  # axis aggregation: "mean"|"max"|"l2"
    # Tier 2
    population_window: int = 80         # covariance window, in ticks
    min_population: int = 30            # sessions required before scoring
    winsor_pct: float | None = 0.99     # per-axis clip percentile; None disables
    winsor_k: int = 3                    # quantile lookback = window × winsor_k ticks
```

```python
@dataclass(frozen=True)
class QueueConfig:
    top_k: int = 200          # stratum 1: highest d²
    flagged: int = 150        # stratum 2: from AMDM-flagged ticks
    per_axis: int = 100       # stratum 3: split evenly across axes, worst per axis
    uniform: int = 100        # stratum 4: uniform random (unknown unknowns)
    reserved: int = 50        # stratum 5: analyst ad-hoc budget (capacity reserved,
                              #   library never fills it)
    seed: int = 0             # reproducible random strata
```

**Validation (constructor):** A ≥ 2; M_a ≥ 1 per axis; 0 < λ ≤ 1; w ≥ 2;
min_warm ≥ 2; population_window ≥ 1; strata ≥ 0. Axes/metrics may differ per
user — the topology is **fixed at construction** and stored in the snapshot.

### 3.2 State (the checkpointable value)

```python
@dataclass(frozen=True)
class MonitorState:
    config: MonitorConfig
    tick: int                                    # global tick counter (0-based)
    # Tier 1, per metric i: ring buffer of length w + per-metric valid count
    metric_buf: np.ndarray        # (A, M_max, w) float64, read-only
    metric_count: np.ndarray     # (A, M_max) int64, read-only
    # Tier 1, per axis
    theta: np.ndarray | None      # (A,) EWMA baselines; None until first S_A
    axis_buf: np.ndarray          # (A, w) ring of S_A history
    axis_count: np.ndarray        # (A,)
    # Joint (Welford)
    n: int
    mu: np.ndarray                # (A,)
    M2: np.ndarray                # (A, A) scatter
    # Tier 2 population: sessions retained for retention =
    # max(population_window, window × winsor_k) ticks — covariance uses the
    # last population_window ticks; winsor quantiles use the full retention
    population: SessionWindow     # see 3.4; immutable, read-only arrays
```

### 3.3 Inputs and results

```python
@dataclass(frozen=True)
class TickMetrics:        # what the caller hands `advance` each tick
    X: np.ndarray        # ragged-safe: per-axis arrays, total A entries

@dataclass(frozen=True)
class SessionFeatures:    # Tier 2 candidate sessions, this tick
    ids: np.ndarray      # (S,) string/int ids
    tick: int            # tick these sessions belong to
    F: np.ndarray        # (S, A) per-axis session feature values, raw scale

@dataclass(frozen=True)
class AxisResult:
    axis: str; S_A: float | None; theta: float | None
    sigma_S: float | None; deviation: float | None
    threshold: float | None; flagged: bool; note: str

@dataclass(frozen=True)
class JointResult:
    n: int; D2: float | None; threshold: float
    p_value: float | None; flagged: bool; note: str

@dataclass(frozen=True)
class TickResult:
    tick: int
    axis_results: tuple[AxisResult, ...]
    joint: JointResult
    S_vector: np.ndarray | None      # (A,) axis scores, None during axis warm-up
    any_axis_flag: bool
    joint_flag: bool
    alert: bool

@dataclass(frozen=True)
class SessionScore:
    id: str; d2: float | None; p_value: float | None

@dataclass(frozen=True)
class ReviewQueuePlan:
    tick: int
    strata: Mapping[str, tuple[str, ...]]   # stratum name → session ids (disjoint)
    unallocated: tuple[str, ...]            # candidates not selected
```

### 3.4 Parquet checkpoint schema

A checkpoint is a **directory** (atomic write: temp dir + rename), versioned.

`state.parquet` — single row, nested columns (pyarrow):
```
format_version  : int64          (currently 1)
config_json     : string         (MonitorConfig, canonical JSON)
tick            : int64
metric_buf      : list<list<list<double>>>   (A × M_max × w, row-major)
metric_count    : list<list<int64>>
theta           : list<double> | null
axis_buf        : list<list<double>>         (A × w)
axis_count      : list<int64>
joint_n         : int64
joint_mu        : list<double>
joint_M2        : list<list<double>>
```

`population.parquet` — tabular, one row per retained session:
```
session_id      : string
tick            : int64
features        : list<double>   (length A)
tick_flagged    : bool           # was the session's tick AMDM-flagged?
absorbed_at     : int64          # tick at which the row entered the window
```

`load_checkpoint(path) -> MonitorState` revalidates `format_version` and config
shape against the arrays, and marks all arrays read-only.

## 4. Component/Module Breakdown (API Definitions)

```
src/agent_monitor/
├── __init__.py        # public exports
├── types.py           # frozen dataclasses (§3) — zero logic
├── config.py          # config validation, k default = χ²_A(0.99), JSON codec
├── agg.py             # axis aggregation registry: {"mean","max","l2"} → pure fns
├── normalizer.py      # Step 1: rolling z-scores over ring buffers
├── axis.py            # Steps 2–4: S_A, EWMA θ_A, per-axis flag
├── joint.py           # Steps 5–6: Welford, Mahalanobis D², χ² test  [CRITICAL]
├── population.py      # Tier 2: session window, empirical (μ̂, Σ̂⁻¹), d²  [CRITICAL]
├── queue.py           # review queue stratification                   [CRITICAL]
├── checkpoint.py      # Parquet save/load (only module with I/O)
└── monitor.py         # thin stateful facade + streaming loop helper
```

All functions below are **pure** unless marked *(I/O)*.

### normalizer.py — Step 1
```python
def zscore(buf: np.ndarray, count: int, x: float) -> tuple[float | None, np.ndarray, int]:
    """Append x to the ring buffer (returning a NEW read-only buffer + count);
    return None until ≥2 samples; return 0.0 if σ < 1e-10 (constant metric)."""
```

### axis.py — Steps 2–4
```python
def update_axis(state: MonitorState, a: int, X_a: np.ndarray,
                agg_fn: Callable) -> tuple[MonitorState, AxisResult]
# S_A = agg(z-scores of metrics with count ≥ 2); θ_A EWMA cold-starts at first S_A;
# flag iff σ_{S_A} > 1e-10 and |S_A − θ_A| > k·σ_{S_A}  (σ over full S_A window).
```

### joint.py — Steps 5–6  *(mutmut target)*
```python
def welford_update(n, mu, M2, S) -> tuple[int, np.ndarray, np.ndarray]
    # rank-one mean + scatter update, no lookahead (S of tick t only).

def d_squared(S, mu, sigma_inv) -> float
    # (S−μ)ᵀ Σ⁻¹ (S−μ); ridge-regularized Σ = M2/(n−1) + ridge·I, full inversion
    # (A ≤ ~10 → O(A³) per tick is fine; Sherman–Morrison is a documented v1 option).

def update_joint(state, S_vector) -> tuple[MonitorState, JointResult]
    # Returns warming-up result until n ≥ min_warm; then D², p = sf(D², A),
    # flagged = D² > χ²_A(1−α).
```

### population.py — Tier 2  *(mutmut target)*
```python
def absorb(state, sessions: SessionFeatures, tick_flagged: bool) -> MonitorState
    # Append rows (with dedup on session_id); evict rows older than
    # retention = max(population_window, window × winsor_k) ticks.
    # Pure: returns new SessionWindow.

def winsor_caps(state) -> np.ndarray | None
    # Per-axis two-sided clip bounds [q_{1−p}, q_p], p = winsor_pct, over the
    # FULL retention window (long lookback → stable caps). None while
    # winsor_pct is None (disabled) or retained rows < 100 (warm-up).

def population_stats(state) -> tuple[np.ndarray, np.ndarray, int] | None
    # (μ̂, Σ̂⁻¹, n) over the last population_window ticks of *winsorized* rows
    # (clipped to winsor_caps; raw rows while caps are None). None until
    # n ≥ min_population. Lazy exact computation from the window (no
    # incremental scatter state). Σ̂ = cov(clipped rows) + ridge·I;
    # Σ̂⁻¹ via np.linalg.inv; raises if singular.

def score_sessions(state, sessions: SessionFeatures) -> tuple[SessionScore, ...]
    # d² per session vs population stats. Candidates are scored RAW — never
    # clipped — so extreme offenders retain extreme d². None during warm-up.
```

### queue.py — review allocation  *(mutmut target)*
```python
def build_review_queue(tick_result, session_scores, state, qcfg: QueueConfig
                       ) -> ReviewQueuePlan
# Strata, filled in priority order with disjoint membership (a session selected
# by an earlier stratum is invisible to later ones):
#   1. top_k      — highest d² among candidates with d² ≠ None
#   2. flagged    — sessions (from the population window + current tick) whose
#                   tick had tick_flagged, random order
#   3. per_axis   — floor(per_axis / A) per axis: max |z of that axis feature|
#                   (feature vs μ̂, σ̂_a diag), most extreme first
#   4. uniform    — seeded RNG (default_rng(qcfg.seed)) uniform draw from remainder
#   5. reserved   — never filled; reduces effective candidate pool by `reserved`
# Each stratum takes min(budget, available); shortfalls cascade to the next
# stratum only within the same build call. Candidates = current tick's sessions.
```

### checkpoint.py *(I/O)*
```python
def save_checkpoint(state: MonitorState, path: Path) -> None    # temp dir + rename
def load_checkpoint(path: Path) -> MonitorState                # validates version
```

### monitor.py — facade (the only stateful class)
```python
class Monitor:
    """Holds the current MonitorState; delegates to the pure core."""
    def __init__(self, config: MonitorConfig): ...
    @classmethod
    def restore(cls, path: Path) -> Monitor: ...
    def update(self, tick: TickMetrics) -> TickResult          # advance + rebind
    def absorb(self, sessions: SessionFeatures) -> None
    def score(self, sessions: SessionFeatures) -> tuple[SessionScore, ...]
    def review_queue(self, tick_result, sessions, qcfg) -> ReviewQueuePlan
    def save(self, path: Path) -> None
    # convenience: update() records tick_flagged on sessions absorbed this tick.
```

**Top-level orchestration (pure):**
```python
def advance(state: MonitorState, tick: TickMetrics) -> tuple[MonitorState, TickResult]
    # axis updates → S_vector (None entries → 0.0 fill during axis warm-up,
    #  matching the reference implementation; see §8) → joint update → TickResult
```

## 5. Step-by-Step Implementation Roadmap

**Phase 1 — Foundations**
- **1a. Types & config** (`types.py`, `config.py`): all §3 dataclasses;
  `MonitorConfig`/`QueueConfig` validation; canonical JSON codec.
  *Done when:* config validation unit tests pass; mypy strict clean.
  `depends_on: []`
- **1b. Aggregation registry** (`agg.py`): `mean`, `max`, `l2` pure functions
  over variable-length z-score lists. *Done when:* each agg has unit tests
  incl. empty/1-element lists. `depends_on: [1a]`

**Phase 2 — Tier 1 AMDM core**
- **2a. Normalizer** (`normalizer.py`): ring-buffer z-scores, constant-metric
  guard, warm-up None semantics. *Done when:* property tests (hypothesis):
  z of iid N(0,1) has |mean| < ε, σ ≈ 1 over large windows; window eviction
  exact. `depends_on: [1a]`
- **2b. Axis monitor** (`axis.py`): S_A, EWMA θ_A cold start, σ_{S_A}, flag.
  *Done when:* unit tests for each step; flag only after σ_{S_A} available.
  `depends_on: [1a, 1b, 2a]`
- **2c. Joint detector** (`joint.py`): Welford, ridge inverse, D², χ² test,
  warm-up. *Done when:* Welford matches `np.cov` on identical inputs (oracle);
  D² ≥ 0 always; p ∈ [0,1]. `depends_on: [1a]`
- **2d. `advance` + reference parity** (`__init__.py`, tests): stream the three
  reference scenarios (normal → coordinated +1.5σ shift → single-axis ×20σ
  spike) through both `references/AMDM.py` and our `advance` on identical
  seeded inputs. *Done when:* S_A, θ, D², and all flags match the reference to
  1e-9 (where reference behavior is defined, incl. 0.0-fill warm-up).
  `depends_on: [2a, 2b, 2c]`

**Phase 3 — Tier 2 sessions**
- **3a. Population window & stats** (`population.py`): absorb/evict (retention
  spans the winsor lookback), two-sided winsor caps, exact empirical (μ̂, Σ̂⁻¹)
  from the clipped window, singularity guard. *Done when:* eviction bound
  holds under property tests; winsorized stats match the clip-then-cov oracle
  (explicit clip → `np.mean`/`np.cov`); caps are None during warm-up.
  `depends_on: [1a]`
- **3b. Session scoring**: d² and p-values for candidate batches.
  *Done when:* batch d² matches per-session loop exactly; warm-up returns None
  scores. `depends_on: [3a, 2c]`
- **3c. Review queue** (`queue.py`): five strata, disjoint allocation, seeded
  RNG, cascade shortfalls. *Done when:* accounting invariant tests pass (§6);
  determinism test (same seed → same plan). `depends_on: [3b]`

**Phase 4 — Persistence & facade**
- **4a. Parquet checkpoint** (`checkpoint.py`): nested state schema +
  population table, atomic directory write, version validation.
  *Done when:* save→load round-trip yields byte-identical arrays and identical
  subsequent results (integration test). `depends_on: [1a, 2*, 3* schemas]`
- **4b. Monitor facade + streaming helper** (`monitor.py`): stateful wrapper,
  restore/save, tick_flagged bookkeeping. *Done when:* facade delegation tests;
  an Airflow-style "load → advance → score → save" usage documented in README.
  `depends_on: [4a]`

**Phase 5 — Hardening & release**
- **5a. Property-based soak tests** (hypothesis): no-lookahead equivalence
  (streaming vs batch replay), long-run H₀ behavior (mean D² ≈ A, flag rate ≈ α
  after warm-up), NaN/Inf-free outputs under adversarial inputs.
  `depends_on: [2d, 3c]`
- **5b. Mutation testing (restricted)**: `mutmut` on `joint.py`,
  `population.py`, `queue.py` only; all generated mutants killed or explicitly
  justified. `depends_on: [5a]`
- **5c. Release hygiene**: README with streaming + Airflow examples, coverage
  ≥ 85%, ruff/mypy clean, uv lockfile. `depends_on: [5a, 5b, 4b]`

## 6. System & Test Invariants

**Temporal (No Lookahead)**
- **[T1] [ATOMIC]** `result(t)` and `state(t)` are functions of ticks
  `0..t` only. *Test:* streaming `advance` sequence ≡ batch recompute over the
  full prefix, to 1e-12.
- **[T2] [ATOMIC]** EWMA θ and Welford (n, μ, M2) at tick t never incorporate
  S(t+1) or later. *Test:* mutation-verified in `joint.py`.

**Oracles (independent references)**
- **[O1] [ATOMIC]** Welford scatter ≡ `np.cov(ddof=1)` and Welford mean ≡
  `np.mean` on identical data.
- **[O2] [ATOMIC]** Population (μ̂, Σ̂) ≡ clip-then-cov oracle: `np.mean`/
  `np.cov` over retained window rows clipped to `winsor_caps` (raw rows while
  caps are None).
- **[O3] [INTEGRATION]** Full-pipeline parity with `references/AMDM.py` on the
  three seeded scenarios, including the detection matrix: normal → silent;
  coordinated +1.5σ shift → joint-only flag; single-axis ×20σ spike → both
  layers flag.

**Accounting**
- **[A1] [ATOMIC]** Review queue: strata are pairwise disjoint;
  `|stratum| = min(budget, available_after_higher_priority_strata)`; no session
  appears twice; `reserved` is never filled by the library. Oracle: QueueConfig
  vs plan sizes.
- **[A2] [ATOMIC]** Population retention: `rows_retained ≤ capacity`; no row
  with `tick < state.tick − retention + 1` survives an `absorb`, where
  `retention = max(population_window, window × winsor_k)`; the covariance
  window is the last `population_window` ticks of that.

**Plausibility**
- **[P1] [ATOMIC]** Under iid N(0,1) metric inputs, after warm-up:
  E[D²] ≈ A (χ²_A), empirical flag rate ≈ α within tolerance over ≥ 10⁴ ticks
  (hypothesis/Monte-Carlo with fixed seeds).
- **[P2] [ATOMIC]** All emitted floats are finite: no NaN/Inf in any result
  under any valid input (incl. constant metrics, zero-variance axes, ridge
  singular case).
- **[P3] [ATOMIC]** Tier 2 H₀ plausibility: sessions sampled iid from the
  population have mean d² ≈ A; empirical flag rate lies within [α, 2α] over
  ≥ 10⁴ Monte-Carlo sessions. The band above nominal α is the documented
  variance-shrinkage effect of winsorization (~2–3% at p=0.99); exact
  calibration is deferred to v1.
- **[P4] [ATOMIC]** Candidates are never clipped: a candidate feature beyond a
  winsor cap scores strictly greater d² than one sitting at the cap (no
  censoring of the extremes the top-k stratum must catch).

**Determinism & round-trip**
- **[D1] [ATOMIC]** Same (state, seed, inputs) → identical outputs; random
  strata reproducible from `QueueConfig.seed`.
- **[D2] [INTEGRATION]** `load(save(state)) ≡ state` (arrays byte-equal,
  read-only), and the next `advance` produces identical results either way.

**Realism gate**
- **[R1]** [INTEGRATION] tests under 2d/5a use synthetic streams that exercise
  the full lifecycle (warm-up → steady state → drift → spike → recovery) with
  realistic tick cadence, not single-shot mocks. Calibration against *real*
  production telemetry is explicitly deferred (v1) — defaults are the paper's.

## 7. Known Assumptions

1. **Caller-aligned inputs (Q4a).** Each tick delivers one complete A×M matrix;
   the caller is responsible for joining raw-log and LLM-judge data (the
   30-min-cadence alignment). Partial/late-arriving axes are a v1 extension.
2. **Session features are caller-supplied** raw per-axis values (length A per
   session); the library does not derive or normalize them beyond the empirical
   population transform.
3. **Topology fixed at construction** (axes, metric names, w, λ, k, α); changes
   require a new state.
4. **Defaults follow the paper**: w=80, λ=0.25, k=χ²_A(0.99), α=0.01,
   min_warm=30, ridge=1e-6, mean aggregation. Aggregation is pluggable via
   name in config.
5. **No alerting/notifications** — flags and plans are return values; delivery
   is downstream.
6. **No metric encoding guidance** — categorical/boolean metrics are the
   caller's responsibility (e.g., "fraction < 0.5").
7. **Checkpoint security**: Parquet only (no pickle) — loading a checkpoint is
   data-only, no code execution; `format_version` gates schema evolution.
8. **QueueConfig budgets are per build call.** The 200/150/100/100/50 defaults
   assume one build per day; users running per-tick builds must scale them.
9. **Joint detector consumes 0.0-filled S entries during per-axis warm-up**
   (reference parity); negligible for min_warm ≥ 30 (see §8 risk R2).
10. **Winsorization is on by default** (`winsor_pct=0.99`, two-sided, quantile
    lookback `window × 3` ticks) to protect population estimates from
    contaminated sessions; it mildly inflates Tier 2 d² under H₀ (variance
    shrinkage), so nominal p-values are approximate — exact calibration is a
    v1 task with real data. It does not protect against *coordinated* drift
    (all sessions shifting together), which no rolling-window scheme can
    identify without an external anchor.

## 8. Potential Edge Cases & Pre-Mortem Risks

| # | Risk / edge case | Mitigation |
|---|---|---|
| R1 | **Singular / near-singular covariance** (perfectly correlated axes, degenerate population) | ridge ε·I on both Σ̂; explicit `LinAlgError` → informative error; property test [P2] |
| R2 | **0.0-fill during axis warm-up biases early joint μ/Σ** | Documented reference-parity choice; axis warm-up ≤ a few ticks vs min_warm=30; revisit in v1 |
| R3 | **Constant metric (σ=0)** or **σ_{S_A}=0 axis** | z := 0.0; flag suppressed for that axis (matches reference guards) |
| R4 | **Alert storms / flag thrashing** under marginal drift | EWMA θ absorbs regime shifts by design; k defaults intentionally strict (≈ χ²₅(0.99)); true fix = calibration, deferred to v1 with explicit TODO |
| R5 | **Population contamination**: anomalies absorbed into (μ̂, Σ̂) shift the baseline and inflate Σ̂ (masking) | Mitigated by two-sided winsorization: per-axis clip at `winsor_pct` (default p99), quantiles estimated over the longer `window × winsor_k` lookback (stable caps, responsive covariance). Candidates are never clipped, so extremes still score extreme ([P4]). Residual: coordinated drift is inherently un-protectable (see Assumption 10); flagged ticks are recorded for downstream forensics |
| R6 | **Duplicate session ids** across ticks | `absorb` dedups on (session_id); documented |
| R7 | **Checkpoint schema evolution** breaks old snapshots | `format_version` column; loaders reject unknown versions loudly |
| R8 | **Partial checkpoint write** (crash mid-save) | Atomic temp-dir + rename in `save_checkpoint` |
| R9 | **Huge candidate batches** (10⁴–10⁷ sessions) overwhelm queue build | All vectorized (d² batched via `(F−μ̂) @ Σ̂⁻¹`); O(S·A²); no per-session Python loops |
| R10 | **Timezone / tick-alignment drift** | Out of scope by A1; tick index is an integer sequence, wall-clock semantics belong to the caller |
| R11 | **min_population never reached** (low-volume deployments) | `score_sessions` returns None scores + explicit `note`; no silent zeros |
| R12 | **Aggregation function misconfiguration** (unknown `agg` name) | Validated at config construction; registry lookup fails fast |
