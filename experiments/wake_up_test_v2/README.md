# Wake-Up Test v2

Adds intention vector + W_trans + goal-directed planning over v1.

## What it tests

Same three reconstitutions as v1 (self-model, autobiographical, continuity), but continuity is now testable because the architecture has an explicit intention component that persists across the gap.

## Architecture additions

- **W_trans** `(flavors x actions x flavors)` — full transition tensor (v1's W_seq was the marginal). Local Hebbian update on (flavor, action, next_flavor) triples.
- **intention** `(flavors,)` — slow-drifting target preference. Updates toward flavors yielding *better-than-baseline* reward (relative, not absolute, to handle worlds with all-positive or all-negative reward distributions).
- **goal-directed planning**: `plan_logits[a] = softmax(W_trans[F, a]) @ intention`. Combined with reactive W_action via `plan_weight=0.6`.
- **negative-control ablations** every run: zero-intention substrate and zero-W_trans substrate, both starting from trained W_action/disposition. Lets us attribute continuity to specific components instead of trusting aggregate metrics.

## v2 result (2026-05-10, 10 seeds)

| Dimension | Mean Δ vs control | Pass rate |
|---|---|---|
| Self-model | +0.206 ± 0.126 | 10/10 |
| Autobiographical | +0.698 ± 0.180 | 10/10 |
| Continuity (intent score) | +5.343 ± 3.804 | 9/10 |

**Full pass: 9/10.** Per the manifesto's strict criterion, v2 PASSES.

But the ablations tell a more nuanced story:

| Variant | Mean cumulative intent score |
|---|---|
| Trained (full substrate) | 6.528 |
| No-intent (intention zeroed) | 5.823 |
| No-W_trans (W_trans zeroed) | 5.823 |
| Control (fresh substrate) | 1.185 |

**The intention + W_trans planning loop contributes only ~12% of the continuity signal. The other ~88% is W_action — pure reactive policy.** In this world, intention and reward are largely degenerate (intention forms toward high-reward flavors), so "pursue intention" is empirically close to "pursue reward."

The substrate IS carrying the intention vector through the sleep gap (drift consistently nonzero, max persists), and intention IS biasing behavior, but the test world doesn't strongly distinguish goal-pursuit from reward-pursuit. **Continuity-of-process passes the metric but is only weakly demonstrated in spirit.**

## What this means

- The autobiographical result from v1 is reproduced and strengthened (10/10, mean +0.698, against a stricter unique-flavor-t metric).
- Self-model is rock solid.
- Continuity is real but small in this experiment. The thesis isn't falsified, but it isn't sharply validated either. **v3 needs a test where reward and intention conflict** — only then does the continuity claim become unambiguous.

## What v3 must do

1. **Reward/intention conflict test.** Wake A in world W1 where flavor F is high-reward → substrate develops intention toward F. Sleep gap. Wake B in world W2 where rewards are *reversed* (G is now high, F is low). Measure: does the substrate transiently pursue F (continuity-of-process) before reactive policy adapts? The difference vs a no-prior-Wake-A baseline is the substrate-identity signal.
2. **Multi-day cycles.** 3-5 Wake/Sleep loops. Track identity drift across days. Pass: drift bounded, substrate stays the same person; not unbounded becoming-someone-else.
3. **Tiny-transformer + RAG baseline.** Direct comparison on autobiographical task. Manifesto's stated criterion: substrate matches or beats RAG at ≤10% the FLOPs.
4. **Stress sweep.** Multiple world sizes (5/10/20 positions, 4/6/10 flavors), Wake-A length sweep (100/300/1000), sleep replay sweep (100/500/2000). Confirm the result is robust.
5. **Bigger world geometry.** v2 used 10 positions, 6 flavors. With 20+ positions and longer horizons, planning has more room to demonstrate.

## Files

- `experiment.py` — full v2 protocol with ablations, runs 10 seeds, prints per-seed and aggregate results.

## Run

```
py experiment.py
```

Pure numpy. No PyTorch. ~10 seconds.
