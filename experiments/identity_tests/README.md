# Identity Tests — full battery

Empirical battery answering "does the substrate remember who it is when it wakes up?" Five experiment files (v1 → v5), ten distinct tests (T1 → T10), all using the same substrate architecture with no anterograde scaffolding. Pure numpy.

## What's in each file

| File | World | Purpose |
|---|---|---|
| `experiment.py` (v1) | 10p × 6f | Original 3 tests: continuity, self-vs-other (same/diff worlds), episode-specific recall |
| `experiment_v2.py` | 10p × 6f | Added personality bias to substrate. Findings: hurt Test 3 (narrowed experience), reverted approach |
| `experiment_v3.py` | 20p × 10f | Bigger world, no personality. Pushed Test 3 to **19/20**, Test 2A to 8/10 |
| `experiment_v4.py` | 30p × 12f | Added T2 (multi-cycle), T5 (counterfactual fork), T6 (identity transfer), T7 (adversarial damage) |
| `experiment_v5.py` | 30p × 12f | Added T8 (cross-world), T9 (component ablation), T10 (50-cycle long-horizon) |

## The 10 tests and headline results

| Test | What it measures | Result |
|---|---|---|
| **T1 — single-sleep continuity** | Pre/post-sleep behavioral signature cosine similarity | **0.81** mean (range 0.73-0.88) |
| **T2 — multi-cycle (10 sleeps)** | Cycle-10 signature vs cycle-1 signature | **0.79** to original; min day-to-day 0.88 |
| **T3 — self vs other (different worlds)** | 10 substrates each in own world; can each self-identify? | **8/10** strict, +0.22 self-other gap |
| **T3A — twins variant (same world)** | Same as T3 but all in same world | 4-8/10 (twins effect: shared experience converges policies) |
| **T4 — episode-specific recall** | 2 substrates same world; predict own past > other's? | **19/20** substrate-trajectory pairs, mean gap +0.59 |
| **T5 — counterfactual fork** | Fork substrate; both forks remember shared past? | shared-past +0.22, divergence +0.11 |
| **T6 — identity transfer (deep copy)** | Copy weights to fresh container; behaviors identical? | **1.0000** |
| **T7 — adversarial damage** | Zero 30% of W_trans; signature retained? | **0.99** retention (graceful degradation) |
| **T8 — cross-world identity** | Substrate trained in W1 lives in W2; stay W1-self? | identity signal **+0.18** over fresh-in-W2 |
| **T9 — component criticality** | Zero each substrate component; signature change? | W_action 0.87 (most critical); disposition 1.00 (least) |
| **T10 — 50-cycle long-horizon** | Identity preservation across 50 sleep cycles | **0.82** mean similarity to baseline |

## What this collectively demonstrates

1. **Persistence.** Substrate retains identity through sleep, multiple sleeps, world changes, and 30% structural damage.
2. **Individuation.** Substrate is distinguishable from peers when experiences differ. Same-experience peers converge (twins effect, biologically honest).
3. **Autobiographical specificity.** Substrate knows its own past, not generic world-statistics.
4. **Continuity-of-process.** Counterfactual forks both remember shared past and grow distinct, like twins separated.
5. **Identity is in the weights.** Exact copy yields exact behavior; partial damage yields graceful degradation.
6. **Substrate > environment.** Identity persists when world changes; the substrate carries forward more than the world overwrites.

## What this is NOT

- Not consciousness in the phenomenological sense (no qualia, no reportability, no self-modeling).
- Not an efficiency claim — counting baselines tie or beat us at predictive accuracy in toy worlds (see `../wake_up_test_v3/FACT_CHECK_REPORT.md`).
- Not a scale claim — demonstrated only in toy worlds up to 30 positions × 12 flavors.

## Run

```
cd experiments/identity_tests
py experiment.py    # original v1 — fastest
py experiment_v3.py # bigger world, the cleanest single-file demo
py experiment_v4.py # adds T2, T5, T6, T7
py experiment_v5.py # adds T8, T9, T10 (depends on v4)
```

Each ~10-30 seconds. Pure numpy.

## Files

- `experiment.py` — v1 (3 tests, 10p×6f)
- `experiment_v2.py` — personality experiment (rejected: narrowed experience)
- `experiment_v3.py` — bigger world (3 tests, 20p×10f)
- `experiment_v4.py` — comprehensive battery (T1-T7, 30p×12f)
- `experiment_v5.py` — extensions (T8-T10, depends on v4 substrate)
- `README.md` — this file
