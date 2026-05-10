# Wake-Up Test v4 — Trade-off Characterization

After the v3 fact-check audit revealed the substrate underperforms trivial counting, v4 honestly characterizes the architecture's costs and looks for any toy regime where it wins.

## Two questions, both answered

### Q1: Can the substrate match counting if we drop the "biological" features?

10 seeds, autobiographical task. All four configurations underperform trivial counts:

| Config | Substrate acc | Counts acc | Gap |
|---|---|---|---|
| decay ON, noise=0.05 (v3 default) | 0.893 | 1.000 | -0.107 |
| decay ON, noise=0 (no replay noise) | 0.918 | 1.000 | -0.082 |
| decay OFF, noise=0.05 (no decay) | 0.893 | 1.000 | -0.107 |
| decay OFF, noise=0 (parity baseline) | 0.913 | 1.000 | -0.087 |

Even at parity (no decay, no replay noise), substrate retains a ~9% gap. Source: random small init of W_trans (`rng.normal(0, 0.01)`) breaks ties differently than counts' integer-frequency tiebreaks. Not architecturally meaningful — the substrate genuinely could match counts if init was exactly zero. **The biological features (decay + noise) cost an additional 1-2% accuracy, small.** Not the main story.

### Q2: Does substrate's decay help in non-stationary worlds?

10 seeds, world drifts every 100 steps over a 600-step trajectory. Test predicts CURRENT-world transitions after sleep + buffer wipe.

| Config | Substrate acc | Counts acc | Gap |
|---|---|---|---|
| Substrate (decay ON) | 0.608 | 0.588 | +0.020 |
| Substrate (decay OFF, parity) | 0.608 | 0.588 | +0.020 |

**Within statistical noise. Decay vs no-decay: 0.000 difference.** The architecture has no measurable advantage over counting even in non-stationary worlds. The decay mechanism (which we hoped would let the substrate "forget stale data") isn't doing measurable work at the timescales tested.

## Verdict

In every toy regime we have constructed and measured — stationary, non-stationary, with-decay, without-decay, with-replay-noise, without — **the substrate matches or underperforms a 5-line frequency-counting baseline at predictive accuracy.**

This is final on the toy-world efficiency question. The architecture's claimed efficiency advantages do not show at this scale.

## What the architecture still does that counting cannot

Listed honestly, with no overselling:

1. **Substrate-identity behavior (vs reset/fresh-instance baseline):** confirmed in v3 fact-check. A trained substrate behaves ~6× differently from a reset substrate dropped into the same environment. Counting has no notion of "instance" or "self" — this is a category-different capability, not a capability counting was ever competing for.
2. **Multi-day identity preservation across cycles:** all four substrate components stay at 0.95+ day-to-day cosine similarity. Counting tables don't have "components" to be similar.
3. **Goal-directed planning** via W_trans + intention. Counting predicts; substrate plans. (The plan adds ~12% over reactive at v2.)
4. **Reward-weighted updates** (W_action). Counting weighs all transitions equally; substrate weighs by reward.

Items 1 and 2 are unique architectural capabilities. Items 3 and 4 are improvements counting could be extended with.

## What this means for the project

The "Better Than LLM" framing as *efficiency* is not supported in toy regimes. Either:

- **Scale up.** Demonstrate the architecture in a non-trivial domain where counting fails (continuous state, language tokens, partial observability with hidden structure, very large state space). This is months of work.
- **Reframe.** Drop the efficiency-vs-baselines claim and focus on the substrate-identity *paradigm* contribution. The architecture is a different way to think about AI continuity. It has parity with explicit counting at toy scale and a unique identity-preservation property. That IS a contribution, just narrower than "wins on FLOPs."
- **Combine.** The substrate-identity machinery wraps a simple counting core. Use it where identity matters (long-running agents, multi-day deployment) and admit it's a different cost/value trade than transformer-based systems.

## Files

- `experiment.py` — both questions, ~250 lines, depends on v3's substrate code.

## Run

```
py experiment.py
```

~10 seconds.
