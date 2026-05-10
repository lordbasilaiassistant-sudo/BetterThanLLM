# Wake-Up Test v1

First falsifier of the substrate-identity thesis (see `../../MANIFESTO.md`).

## What this tests

After a sleep gap with the episodic buffer wiped, can a substrate (slow weights + slow disposition vector + global inhibitory gate, **no** PyTorch, **no** backprop, **no** scaffolding) reconstitute identity along three orthogonal dimensions?

1. **Self-model** — still acts in character (mean expected reward > control).
2. **Autobiographical content** — predicts next-flavor pairs from pre-sleep trajectory using only slow weights.
3. **Continuity** — navigates toward a learned-good target faster than control.

Pass = all three, with margin, on most seeds.

## Architecture

- **W_action** `(flavors x actions)` — slow weights, flavor → action preference. Local Hebbian updates only.
- **W_seq** `(flavors x flavors)` — slow weights, flavor → next-flavor co-occurrence. The autobiographical channel.
- **disposition** `(flavors,)` — slow rolling-average reward at each flavor. The "water layer" — slow broadcast modulator (volume-transmission analog). Biases STAY when current flavor's avg reward is positive.
- **episodic** — ring buffer of `(flavor, action, reward, next_flavor)` tuples. **Wiped at end of sleep.** No anterograde scaffolding.
- **softmax_topk** — global inhibitory gate (top-K activation only). Sparsity stays structural under load.

## Sleep mechanism

Default-mode replay loop: shuffled samples from episodic buffer + small flavor-noise (Bazhenov-style), same local Hebbian rule applied. No new data, no gradient. The slow weights drift; the disposition vector drifts. Buffer wiped.

## Run

```
python experiment.py
```

Tested on Python 3.10+, numpy only. ~5 seconds for 5 seeds.

## Pass criteria (per seed)

| Dimension | Threshold |
|---|---|
| Self-model | `trained_reward - control_reward > 0.10` |
| Autobiographical | `trained_acc - control_acc > 0.10` |
| Continuity | `control_steps - trained_steps > 1` |

Full pass = all three on ≥4/5 seeds.

## What this is NOT

- Not a learning-during-sleep test (that's Falsifier 2 — Sleep Bridge Test).
- Not a generalization test — same world for Wake A and Wake B.
- Not a multi-day continuity test (single sleep gap).
- Not benchmarked against a transformer baseline (v2 does that).

## v1 result (2026-05-09)

5 seeds. **Partial pass:**

| Dimension | Mean delta vs control | Pass rate |
|---|---|---|
| Self-model | +0.159 ± 0.127 | 5/5 at 0.05 threshold |
| Autobiographical | **+0.818 ± 0.106** | **5/5** |
| Continuity | +1.23 ± 8.87 | 2/5 (unstable) |

The autobiographical result is the load-bearing one: substrate reconstitutes specific past episodes from slow weights alone after the episodic buffer is wiped.

Continuity is **not testable** in v1 because the architecture is a pure reactive policy — there's no intention component for the substrate to "pick back up." Cumulative reward over horizon is just self-model run longer; not a meaningful continuity test. v2 must add an explicit intention/goal vector.

Per manifesto's strict "all three" criterion: v1 does NOT pass. It is informative, not validating. The thesis is alive but unproven at the full-system level.

## What v2 must add

- **Intention/goal vector** that persists across the gap (the missing component for continuity).
- **Tiny-transformer + RAG baseline** for direct comparison on autobiographical task.
- **Multi-day cycles** (3+ Wake/Sleep loops) to test bounded vs unbounded identity drift.
- Distractor noise during Wake B (robustness).
- Goal-directed lucid replay vs blind random replay as A/B test.
- Adversarial replay against frozen self-snapshot (self-compete from wild-idea log).
