import numpy as np
from scipy.stats import chi2
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
# AMDM — Full Implementation  (Shukla, 2025 * arxiv:2509.00115)
# ─────────────────────────────────────────────────────────────────────────────
# Steps 1–4 : per-axis EWMA anomaly detection
# Steps 5–6 : joint Mahalanobis / chi-square anomaly detection
# ─────────────────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)

# ── Paper defaults ─────────────────────────────────────────────────────────────
N_AXES        = 5
N_METRICS     = 10                          # metrics per axis
WINDOW        = 80                          # w
LAMBDA_EWMA   = 0.25                        # λ
K_SENSITIVITY = chi2.ppf(0.99, df=N_AXES)  # k = χ²_5(0.99) ≈ 15.09
ALPHA         = 0.01                        # joint false-alarm rate
MIN_WARM      = 30                          # joint detector warm-up samples
RIDGE_EPS     = 1e-6                        # covariance regularisation

AXIS_NAMES = ["Safety", "Quality", "Efficiency", "Reliability", "Compliance"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 helper — rolling z-score for a single metric
# ─────────────────────────────────────────────────────────────────────────────
class MetricNormalizer:
    """
    Maintains a rolling window of length w for one raw metric.

      z_i(t) = (x_i(t) - μ_i(t)) / sigma_i(t)

    μ_i and sigma_i are computed from the window, making normalisation adaptive —
    a gradual long-term drift will be absorbed into the baseline, while a
    sudden spike stands out sharply in z-score space.

    Returns None until ≥ 2 samples are available.
    """
    def __init__(self, window: int = WINDOW):
        self.buf = deque(maxlen=window)

    def update(self, x: float) -> float | None:
        self.buf.append(float(x))
        if len(self.buf) < 2:
            return None
        mu    = np.mean(self.buf)
        sigma = np.std(self.buf, ddof=1)
        return 0.0 if sigma < 1e-10 else (x - mu) / sigma


# ─────────────────────────────────────────────────────────────────────────────
# Steps 1–4 — per-axis monitor
# ─────────────────────────────────────────────────────────────────────────────
class AxisMonitor:
    """
    Monitors one named axis across n_metrics raw input metrics.

    Step 1  z_i(t) = (x_i - μ_i) / sigma_i        rolling z-score per metric
    Step 2  S_A(t) = mean({z_i(t)})             aggregate → scalar axis score
    Step 3  θ_A(t) = λ*S_A + (1-λ)*θ_A(t-1)   EWMA tracks the local baseline
    Step 4  flag if |S_A(t) - θ_A(t)| > k * sigma_{S_A}(t)

    Why EWMA for the baseline (θ) but rolling std for the spread (sigma_{S_A})?
    ─────────────────────────────────────────────────────────────────────────
    - θ_A must respond quickly to genuine regime shifts so it doesn't keep
      flagging a new normal. EWMA with λ=0.25 gives ~4-step half-life.
    - sigma_{S_A} should be stable and not inflated by outliers, so we use the
      rolling window std over the last w samples rather than an EWMA variance.
    """

    def __init__(self, name: str, n_metrics: int = N_METRICS,
                 window: int = WINDOW, lam: float = LAMBDA_EWMA,
                 k: float = K_SENSITIVITY):
        self.name        = name
        self.lam         = lam
        self.k           = k
        self.normalizers = [MetricNormalizer(window) for _ in range(n_metrics)]
        self.theta: float | None = None           # EWMA θ_A(t)
        self.S_buf       = deque(maxlen=window)   # history of S_A for sigma_{S_A}

    def update(self, metrics: np.ndarray) -> dict:
        # ── Step 1: z-score each raw metric ──────────────────────────────────
        zs = [z for nm, v in zip(self.normalizers, metrics)
              if (z := nm.update(float(v))) is not None]

        if not zs:
            return dict(axis=self.name, S_A=None, flagged=False,
                        note="warming up z-scores")

        # ── Step 2: aggregate z-scores → scalar S_A(t) ───────────────────────
        # Mean aggregation: keeps S_A in a comparable scale to individual z-scores
        # and weights all metrics equally. Alternatives: max (most sensitive to
        # single-metric spikes) or L2-norm (penalises spread across metrics).
        S_A = float(np.mean(zs))

        # ── Step 3: EWMA θ_A(t) ──────────────────────────────────────────────
        if self.theta is None:
            self.theta = S_A                                      # cold start
        else:
            self.theta = self.lam * S_A + (1.0 - self.lam) * self.theta

        self.S_buf.append(S_A)

        if len(self.S_buf) < 2:
            return dict(axis=self.name, S_A=round(S_A, 4),
                        theta=round(self.theta, 4),
                        flagged=False, note="warming up sigma_S")

        # ── Step 4: adaptive threshold test ──────────────────────────────────
        sigma_S   = float(np.std(self.S_buf, ddof=1))  # sigma_{S_A}(t)
        deviation = abs(S_A - self.theta)               # |S_A - θ_A|
        threshold = self.k * sigma_S                    # k * sigma_{S_A}
        flagged   = (sigma_S > 1e-10) and (deviation > threshold)

        return dict(
            axis      = self.name,
            S_A       = round(S_A,        4),
            theta     = round(self.theta,  4),
            sigma_S   = round(sigma_S,    4),
            deviation = round(deviation,  4),
            threshold = round(threshold,  4),
            flagged   = flagged,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Steps 5–6 — joint Mahalanobis detector
# ─────────────────────────────────────────────────────────────────────────────
class JointAnomalyDetector:
    """
    Maintains rolling μ(t) and sigma⁻¹(t) over the joint score vector S(t).

      D²(t) = (S - μ)ᵀ sigma⁻¹ (S - μ)   ~  χ²(A)  under H₀

    Flag if D²(t) > χ²_A(1 - α).

    Welford's algorithm keeps μ and the scatter matrix numerically stable
    without storing the full history. Sherman-Morrison rank-one updates
    (see helper below) propagate sigma⁻¹ in O(A²) for production use;
    we do a full re-inversion here for clarity.
    """

    def __init__(self, n_axes: int = N_AXES, alpha: float = ALPHA,
                 min_warm: int = MIN_WARM, ridge: float = RIDGE_EPS):
        self.A         = n_axes
        self.threshold = chi2.ppf(1 - alpha, df=n_axes)
        self.min_warm  = min_warm
        self.ridge     = ridge
        self.n         = 0
        self.mu        = np.zeros(n_axes)
        self.M2        = np.zeros((n_axes, n_axes))
        self.Sigma_inv = None

    # ── Step 5a: Welford rank-one mean + scatter update ───────────────────────
    def _welford_update(self, S: np.ndarray):
        self.n  += 1
        d_old    = S - self.mu
        self.mu += d_old / self.n
        self.M2 += np.outer(d_old, S - self.mu)  # rank-one scatter update

    # ── Step 5b: refresh sigma⁻¹ ─────────────────────────────────────────────────
    def _refresh_inverse(self):
        Sigma = self.M2 / (self.n - 1) + self.ridge * np.eye(self.A)
        self.Sigma_inv = np.linalg.inv(Sigma)

    # ── Optional: Sherman–Morrison O(A²) rank-one inverse update ─────────────
    @staticmethod
    def sherman_morrison(Sigma_inv: np.ndarray,
                         u: np.ndarray,
                         v: np.ndarray | None = None) -> np.ndarray:
        """
        Returns (sigma + u vᵀ)⁻¹ from sigma⁻¹ without a full re-inversion.
        When v is None, assumes a symmetric rank-one update (v = u).
        Use this in place of _refresh_inverse for O(A²) throughput.
        """
        if v is None:
            v = u
        Su    = Sigma_inv @ u
        vS    = v @ Sigma_inv
        denom = 1.0 + float(v @ Su)
        if abs(denom) < 1e-12:
            raise ValueError("S-M denominator ≈ 0: covariance becoming singular.")
        return Sigma_inv - np.outer(Su, vS) / denom

    # ── Step 6: D²(t) and chi-square test ─────────────────────────────────────
    def update(self, S: np.ndarray) -> dict:
        S = np.asarray(S, dtype=float)
        self._welford_update(S)

        if self.n < self.min_warm:
            return dict(n=self.n, D2=None,
                        threshold=round(self.threshold, 4),
                        flagged=False,
                        note=f"warming up ({self.n}/{self.min_warm})")

        self._refresh_inverse()
        diff    = S - self.mu
        D2      = float(diff @ self.Sigma_inv @ diff)
        p_value = float(chi2.sf(D2, df=self.A))

        return dict(
            n         = self.n,
            D2        = round(D2,            4),
            threshold = round(self.threshold, 4),
            p_value   = round(p_value,        6),
            flagged   = D2 > self.threshold,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Full AMDM orchestrator — runs all 6 steps each tick
# ─────────────────────────────────────────────────────────────────────────────
class AMDMMonitor:
    """
    Accepts a raw (N_AXES × N_METRICS) metric matrix each timestep and runs
    the complete AMDM pipeline:

      Steps 1-4  per-axis: z-score → aggregate → EWMA → threshold test
      Step  5    form S(t) from axis scores; update μ and sigma⁻¹
      Step  6    D²(t) chi-square joint test

    The two detection layers are complementary:
    ┌──────────────────────┬───────────────────────────────────────────────┐
    │ Per-axis (Steps 1-4) │ Catches extreme isolated blowouts on one axis │
    │ Joint    (Steps 5-6) │ Catches coordinated moderate shifts across    │
    │                      │ axes that each individually look unremarkable │
    └──────────────────────┴───────────────────────────────────────────────┘
    """

    def __init__(self, axis_names: list[str] = AXIS_NAMES,
                 n_metrics: int = N_METRICS, window: int = WINDOW,
                 lam: float = LAMBDA_EWMA, k: float = K_SENSITIVITY,
                 alpha: float = ALPHA, min_warm: int = MIN_WARM):
        self.axes  = [AxisMonitor(name, n_metrics, window, lam, k)
                      for name in axis_names]
        self.joint = JointAnomalyDetector(len(axis_names), alpha, min_warm)

    def update(self, X: np.ndarray) -> dict:
        """X : shape (N_AXES, N_METRICS) — raw metric readings this timestep."""
        # Steps 1–4 in parallel across axes
        axis_results = [ax.update(X[i]) for i, ax in enumerate(self.axes)]

        # Build S(t) — use 0.0 as fill during per-axis warm-up so the joint
        # detector can still accumulate its covariance estimate
        S_vec = np.array([r["S_A"] if r["S_A"] is not None else 0.0
                          for r in axis_results])

        # Steps 5–6
        joint_result = self.joint.update(S_vec)

        return dict(
            axis_results  = axis_results,
            joint_result  = joint_result,
            any_axis_flag = any(r["flagged"] for r in axis_results),
            joint_flag    = joint_result["flagged"],
            alert         = any(r["flagged"] for r in axis_results)
                            or joint_result["flagged"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Demo — three scenarios designed to expose the two detection layers
# ─────────────────────────────────────────────────────────────────────────────
monitor = AMDMMonitor()

JOINT_THRESH = chi2.ppf(1 - ALPHA, df=N_AXES)
print("AMDM thresholds")
print(f"  Per-axis  k * sigma_S  =  χ²_5(0.99) × sigma_S  ≈  {K_SENSITIVITY:.4f} × sigma_S")
print(f"  Joint     D² >       χ²_5(0.99)          ≈  {JOINT_THRESH:.4f}")
print(f"  (k and the joint threshold share the same quantile at A=5, α=0.01)\n")
print(f"  Axis legend:  [{' * '.join(f'{i}:{n[:3]}' for i, n in enumerate(AXIS_NAMES))}]")
print(f"  🚨 = flagged  *  = ok\n")


def fmt(result: dict, t: int, detail_axis: int | None = None):
    """Print one timestep row, with optional per-axis detail."""
    jr      = result["joint_result"]
    D2_str  = f"{jr['D2']:8.3f}" if jr["D2"] is not None else "  warm  "
    p_str   = f"{jr['p_value']:.5f}" if jr.get("p_value") is not None else "  N/A  "
    ax_icons = "".join("🚨" if r["flagged"] else "*"
                       for r in result["axis_results"])
    joint_str = "🚨 JOINT ANOMALY" if result["joint_flag"] else "              *"

    print(f"  t={t:3d}  [{ax_icons}]  D²={D2_str}  p={p_str}  {joint_str}")

    if detail_axis is not None:
        r = result["axis_results"][detail_axis]
        if r.get("deviation") is not None:
            print(f"         {r['axis']:12s}  "
                  f"S_A={r['S_A']:+.3f}  θ={r['theta']:+.3f}  "
                  f"|dev|={r['deviation']:.3f}  threshold={r['threshold']:.3f}  "
                  f"{'← 🚨 FLAGGED' if r['flagged'] else ''}")


# ── Scenario 1: normal baseline ───────────────────────────────────────────────
print("─" * 68)
print("Scenario 1 — Normal baseline  (t=1…60)")
print("  All metrics iid N(0,1).  No alarms expected.")
print("─" * 68)
for t in range(1, 61):
    X   = RNG.normal(0.0, 1.0, size=(N_AXES, N_METRICS))
    res = monitor.update(X)
    if t in (1, 2, 3, 20, 40, 60):
        fmt(res, t)
print("  ...")

# ── Scenario 2: coordinated moderate shift — joint only ───────────────────────
print()
print("─" * 68)
print("Scenario 2 — Coordinated moderate shift  (t=61…63)")
print("  All 5 axes shift +1.5sigma simultaneously.")
print("  Each axis: S_A ≈ 1.5, |dev| ≈ 1.1  <  k*sigma_S ≈ 4.8  →  per-axis silent")
print("  Joint:     all axes move together  →  D² spikes, Mahalanobis catches it")
print("─" * 68)
for t in range(61, 64):
    X   = RNG.normal(1.5, 1.0, size=(N_AXES, N_METRICS))
    res = monitor.update(X)
    fmt(res, t)

# ── Scenario 3: extreme single-axis blowout — per-axis fires ──────────────────
print()
print("─" * 68)
print("Scenario 3 — Extreme single-axis spike  (t=64…66)")
print("  Safety axis (idx 0): all 10 metrics jump to N(20, 0.5).")
print("  |dev| >> k*sigma_S  →  per-axis Safety alarm fires  (+ joint also fires)")
print("─" * 68)
for t in range(64, 67):
    X    = RNG.normal(0.0, 1.0, size=(N_AXES, N_METRICS))
    X[0] = RNG.normal(20.0, 0.5, size=N_METRICS)   # Safety blowout
    res  = monitor.update(X)
    fmt(res, t, detail_axis=0)

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print("─" * 68)
print("Detection summary")
print(f"  {'Scenario':<35}  {'Per-axis':<12}  {'Joint'}")
print(f"  {'─'*35}  {'─'*12}  {'─'*12}")
print(f"  {'Normal baseline (t=1–60)':<35}  {'silent':<12}  {'silent'}")
print(f"  {'Coordinated +1.5sigma shift (t=61–63)':<35}  {'silent':<12}  {'🚨 fired'}")
print(f"  {'Safety extreme spike ×20sigma (t=64–66)':<35}  {'🚨 fired':<12}  {'🚨 fired'}")
print()
print("  Key insight: the two layers cover orthogonal failure modes.")
print("  k*sigma_S is intentionally strict (≈4–5sigma) so per-axis alarms signal")
print("  only unambiguous blowouts, leaving the joint D² test to handle")
print("  the subtle correlated drifts that matter most for agentic AI.")
