---
title: "(AMDM) Adaptive Monitoring and Real-World Evaluation of Agentic AI Systems"
authors: 
  - "Shukla, Manish A."
year: 2025
doi: "10.48550/arXiv.2509.00115"
url: "https://arxiv.org/abs/2509.00115"
code_link: null
category: "llm"
tags: 
  - "Agentic AI"
  - "Monitoring"
  - "Evaluation Framework"
  - "Enterprise AI"
  - "Benchmarks"
  - "LLM"
  - "AI"
status: "extracted"
---

## Adaptive Monitoring and Real-World Evaluation of Agentic AI Systems

### 1. The Core Innovation
- The paper introduces the **Adaptive Multi-Dimensional Monitoring (AMDM)** algorithm, a framework for real-time evaluation of autonomous agents. It transitions AI measurement from static, capability-focused benchmarks to a continuous, five-axis monitoring system that adapts to distribution drift in streaming logs.

### 2. Context & Motivation
- A systematic review of 84 papers (2023-2025) reveals that 83% focus solely on technical capability, while only 30% consider economic or human-centered axes. This lack of balanced monitoring leads to undetected "goal drift," hallucinations, and massive cost variations. 
- The paper fills this gap by formalizing an algorithmic approach to catch these operational failures in real-time.

### 3. Methodology

- The prior framework, BASIC, describes what to measure:
1. **Capability + Efficiency:** measure task completion, latency, and resource usage
2. **Robustness + Adaptability:** resilience to noisy inputs, adversarial prompts, and changing goals
3. **Safety + Ethics:** avoidance of toxic or biased outupts and adherence to legal, ethical, policy norms
4. **Human-Centered Interaction:** CSAT, trust, transparency
5. **Economic + Sustainability:** productivity gains, cost per outcome, carbon footprint

**For Methodological/Theoretical Papers:**
- **Notation/Setup**: Let $M(t) = [m_1(t), m_2(t), ..., m_n(t)]^T$ be a vector of heterogeneous metrics at time $t$. The algorithm processes these streaming metrics to detect state deviations. A z-score is therefore $z_i(t) = \frac{m_i(t) - \mu_i(t)}{\sigma_i(t)}$ with $(\mu_i, \sigma_i^2)$ computed over rolling windows of length $w$.
- **Estimands**: We aim to estimate the "Joint Anomaly Score" using the Mahalanobis distance to identify multivariate outliers:
  - Within each of the BASIC axes, we compute an axis score $S_A(t)$. And we use exponential weighted moving averages to detect thresholds
    - $\theta_A(t) = \lambda S_A(t) + (1-\lambda) \theta_A(t-1)$ with smoothing parameters $\lambda \in (0,1]$. An axis is flagged when $|S_A(t) - \theta_A(t)| > k \times \sigma_{S_A,k}$.
  - **Joint Anomaly Score:** $D^2(t) = (S(t) - \mu(t))^T \Sigma(t)^{-1} (S(t) - \mu(t))$ where $S(t)$ represents the normalized axis scores and $\mu(t)$ is the rolling mean.
    - Measures how atypical the joint state is w.r.t to historical data
    - Anomaly occurs due to chi-squared test with 5 d.f.
- **Assumptions**: 
    - **Metric Heterogeneity**: Metrics across different axes (e.g., latency vs. safety) can be normalized into a comparable space.
    - **Temporal Continuity**: Concept drift is gradual enough to be captured by Exponentially Weighted Moving Averages (EWMA).
- **Math Summary**: The AMDM algorithm involves three stages: 1) Normalizing metrics using rolling z-scores, 2) Aggregating them into five weighted axis scores (Capability, Robustness, Safety, Human-Centred, Economic), and 3) Applying adaptive EWMA thresholds to flag anomalies that exceed the historical Mahalanobis distance threshold.

**For Empirical Papers:**
- **Data Source**: The methodology was validated using 300 simulated workflows (software modernization, credit-risk drafting) and 8,400 real-world event logs from a production knowledge-worker assistant.
- **Key Findings**: 
    - Detected "goal drift" (where agents deviate from original instructions) 54% faster than traditional methods.
    - Achieved a high correlation ($\rho=0.83$) with expert-judged production readiness, whereas accuracy-only metrics performed poorly ($\rho=0.41$).

### 4. Implementation Details
- The framework uses **Exponentially Weighted Moving Averages (EWMA)** for thresholding, allowing the system to ignore minor fluctuations while catching significant shifts. The monitoring overhead is highly efficient, adding less than 3% latency to the underlying agent workflow.
- sensitivity must be calibrated before deployment. Recommend monitoring the system during normal operation and then choosing parameters that yield the desired false-positive rate.

```python
"""
Inputs
  - w: window length. Defaults to 80
  - lambda: smooth parameter. Defaults to 0.25
  - k: sensitivity multiplier. Defaults to \chi_5^2(0.99) 
  - alpha: desired joint false alarm rate

Alg:
for t = 1, 2, ... do:
  1. update \mu_i(t), \sigma_i(t), z_i(t)
  2. Aggregate z-scores into per-axis scores S_A(t)
  3. Update EWMAs \theta_A(t)
  4. if |S_A(t) - \theta_A(t)| > k \times \sigma_{S_A}(t) then: flag axis A
  5. Form S(t), update \mu(t) and \Sigma(t)^{-1} using rank-one updates
  6. if D^2(t) > \chi_A^2(1 - \alpha) then: flag joint anomaly
"""
```

### 5. Relation to Other Work
- _See [[CLEAR|(CLEAR) Beyond Accuracy: A Multi-Dimensional Framework for Evaluating Enterprise Agentic AI Systems]]_
- Builds upon the author's prior "Basic" framework by providing the mathematical formalization and empirical evidence missing in previous literature.
- Complements recent benchmarks like **7-bench** or **GAIA** by adding a "Monitoring" layer meant for production rather than just pre-deployment testing.

### 6. Practical Utility & Applicability
- **Compute/Resource Load:** Very low; the Mahalanobis distance calculation and EWMA updates are computationally inexpensive for real-time use.
- **Ease of Implementation:** Medium; requires integration into the agent's logging/telemetry stream.
- **Reproducibility Score:** High; the paper includes a clear algorithmic flow and a reproducibility checklist for practitioners.

### 7. BibTeX
```bibtex
@article{shukla2025adaptive,
  title={Adaptive Monitoring and Real-World Evaluation of Agentic AI Systems},
  author={Shukla, Manish A.},
  journal={arXiv preprint arXiv:2509.00115v3},
  year={2025}
}
```