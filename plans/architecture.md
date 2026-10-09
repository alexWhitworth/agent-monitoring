## Architecture

**Scenario:** Let's say we have 10^7 sessions daily. I assume we're not instrumenting logging on all of these. But we take some small sample 0.1% with full telemetry where we use LLM as a judge for the quality, reliability, and security metrics and cost + latency can be taken from raw logs. Assume we take an even smaller sample of 500-1000 / day for human auditing. 

**Question:** How do we instrument AMDM?



- **Suggested Architecture:** Combine both (Tier 1) aggregate metrics by time unit and (Tier 2) per-session metrics in the cross-section.

```

──────────────────────────────────────────────────────────────

# Overall flow

──────────────────────────────────────────────────────────────

10,000,000 sessions / day

│

├── ALL sessions ──► raw logs ──► cost, latency, error rates

│                                  (Efficiency + Reliability axes, full coverage)

│

└── 0.1% = 10,000 sessions ──► LLM-as-judge ──► quality, safety, compliance

        │                        (~7 sessions/min, delayed)

        │

        ├── TIER 1: Aggregate into time buckets ──► AMDM system monitor

        │           one tick t per bucket (eg. 30 min --> k=80 means 40 hours)

        │           catches: deployment degradation, model drift, infra issues

        │

        └── TIER 2: Per-session Mahalanobis scoring

                    compare each session vs rolling population distribution

                    catches: adversarial inputs, individual edge cases

                    feeds: human review queue prioritization

```



### Tier 1: Aggregate metrics

```

Within one time bucket (e.g. 1 hour):



  ~416 LLM-judged sessions available

  ~416,000 raw log sessions available



For each axis, the 10 "metrics" are now aggregate stats across the sampled LLM-judged and full raw log sessions. eg:



  Quality axis (from LLM judge, n≈416):

    m1 = mean(task_completion_score)

    m2 = p25(coherence_score)

    m3 = p75(coherence_score)       ← spread captures distribution shift too

    m4 = mean(relevance_score)

    m5 = fraction_score < 0.5       ← tail mass is a powerful signal

    ... etc



  Latency axis (from raw logs, n≈416,000):

    m1 = p50/95/99(E2E wall time ms)

    m2 = mean(time to first token)

    m3 = mean(turn latency)

    ... etc

  

  Cost Axis

    m1 = mean(cost_per_session) = costs of input, output, and cache tokens

    ... etc

```



### Tier 2: Session Metrics (prioritizing human review)

```

Let s = 1, 2, ... S index sessions at a fixed time unit

    d_{s}^2 = (x_s - \hat \mu)^T \hat \Simga^{-1} (x_s - \hat \mu)



where (\hat \mu, \hat \Simga^{-1}) come from the Tier 1 computations



Use to allocate 500-1000 sessions to human review



  ┌─────────────────────────────────────┬──────┬──────────────────┐

  │ Stratum                             │  N   │ Purpose                      │

  ├──────────────────────────────┼──────┼─────────────────────────┤

  │ Top-k by session d² score           │  200 │ catch worst individual cases │

  │ Sessions from flagged AMDM buckets  │  150 │ drill into system anomalies  │

  │ Stratified random (per-axis worst)  │  100 │ coverage of each failure mode│

  │ Pure uniform random                 │  100 │ avoid selection bias,        │

  │                                     │      │ **catch unknown unknowns**   │

  │ Reserved for analyst requests       │   50 │ ad hoc investigation         │

  └──────────────────────────────┴──────┴─────────────────────────┘

```



### Timing and Unification

```

# What Each Layer Actually Catches:

──────────────────────────────────────────────────────────────

Failure mode                          │ Tier 1 AMDM │ Tier 2 Session Score

───────────────────────────────┼────────────┼─────────────────

Model version regression              │ ✅ per-axis  │ ✅ elevated d² broadly

Coordinated multi-axis drift          │ ✅ joint D²  │ ⚠️  subtle per session

Single adversarial/jailbreak session  │ ❌ diluted   │ ✅ extreme outlier

Infrastructure latency spike          │ ✅ fast (raw)│ ✅ high latency flag

Slow quality decay over days          │ ✅ EWMA      │ ❌ population shifts too

Novel failure mode (unknown)          │ ❌           │ ❌ → random stratum saves you

```

- Note: T1 and T2 arrive at different times:

    - `t=0:00` time unit starts

    - `t=0:00` Raw logs flush every ~1 minute

    - `t=0:30` LLM as a Judge batch jobs every ~30 minutes. Launch Tier 1 AMDM 

    - `t=0:30` Launch Tier 2 compute

    -  Add flagged session to priority queue for batching to human review
