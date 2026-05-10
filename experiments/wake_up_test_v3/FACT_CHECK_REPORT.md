# v3 Fact-Check Report

Date: 2026-05-10. Honest audit of every headline claim, run before moving on.

## Methodology

For each claim, ran an explicit counter-test designed to falsify it. Counter-tests included trivial baselines we should have included from the start (frequency counting, smart bounded RAG with precomputed table, reset-substrate baseline).

Code: `fact_check.py` in this directory. 10 seeds for autobiographical/conflict, 5 seeds for multi-day decomposition.

## Claim-by-claim verdict

### CLAIM 1 — "Substrate compresses 300 episodes into 36-entry W_trans and predicts at 89% accuracy."

**Verdict: SURVIVES IN ABSOLUTE TERMS, FAILS RELATIVE TO TRIVIAL BASELINES.**

| System | Mean acc | Memory | FLOPs/query |
|---|---|---|---|
| Substrate (W_trans) | 0.893 | 108 floats | 6 |
| Trivial count table (`counts[f,n]+=1`) | **1.000** | 36 ints | 6 |
| Smart bounded RAG (modal next per flavor_t) | **1.000** | ≤6 entries | 1 |
| Frequency baseline (always predict global mode) | 0.462 | 1 int | 1 |

**A 5-line explicit-counting table beats our substrate 4/10 seeds, ties on 6/10.** The substrate is a noisier, slower implementation of frequency counting in this regime. The Hebbian decay (`* (1 - lr*0.05)`) and noisy replay (5% flavor-flip rate) are introducing noise that explicit counting doesn't have.

The architecture's claimed value-add over trivial counting is **not measurable in this toy world.** It would have to be demonstrated in a regime where Hebbian smoothing IS useful — high-dimensional observations, partial observability, distractor noise, longer temporal dependencies.

### CLAIM 2 — "Substrate uses 1% of RAG's FLOPs."

**Verdict: SURVIVES vs unbounded full-RAG, but the comparison is asymmetric.**

- Substrate: 6 FLOPs/query (argmax over F values).
- Full RAG (unbounded, scan all 300 pairs): ~600 FLOPs/query. → substrate is 1%.
- Smart-RAG (precomputed table): **1 FLOP/query (dict lookup)**. → substrate is 6× SLOWER.

The manifesto's "≤10% baseline FLOPs" criterion is satisfied vs full-RAG, which was the manifesto's specified comparison. But framing the substrate as "FLOPs-efficient" is misleading when a smarter baseline exists at 6× lower FLOPs.

### CLAIM 3 — "Bounded RAG only gets 29% accuracy."

**Verdict: TRUE BUT STRAWMAN.**

The 29% number was against FIFO-eviction bounded RAG with K=36 pairs. Smart bounded RAG (one entry per unique flavor_t) gets 100% at smaller memory. We were comparing to a bad baseline. The substrate doesn't actually beat any baseline that bothers to do basic deduplication.

### CLAIM 4 — "Stickiness in conflict test = substrate-identity continuity-of-process."

**Verdict: PARTIALLY OVERSOLD.**

| Variant | Step-30 ref-intent rate |
|---|---|
| A. Trained substrate → W2 (rewards reversed) | +3.388 |
| B. Trained substrate → W3 (totally new world, different geometry) | +2.774 |
| C. Reset substrate → W2 | +0.587 |

- **A vs C (+2.801): substrate-identity vs reset baseline SURVIVES STRONGLY.** A trained substrate behaves very differently from a freshly-reset substrate when both are dropped into W2.
- **A vs B (+0.614): the gap is small.** The trained substrate carries W1-bias into *any* world it's dropped into, not specifically into W2 (the reward-reversed one). The "behavioral inertia under conflict" framing was oversold — what we actually measured was "weights persist after sleep, and persisting weights reflect Wake A's training," which is technically true but architecturally trivial.

### CLAIM 5 — "Multi-day cosine similarity 0.957 reflects identity preservation."

**Verdict: SURVIVES CLEANLY.**

Day-to-day cosine similarity, decomposed by component:

| Component | Mean d2d sim |
|---|---|
| W_action | 0.970 |
| W_trans | 0.958 |
| intention | 0.950 |
| disposition | 0.948 |
| concatenated | 0.957 |

All four components stay similar day-to-day. The 0.957 figure isn't dominated by W_trans converging in a stable world — every part of the substrate is stable. **Identity preservation as measured is real.**

### CLAIM 6 — "Substrate-identity is architecturally distinct from reactive policy."

**Verdict: SURVIVES (this is the strongest result).**

Reset-substrate baseline (Wake A in W1, then *reset all weights* before Wake B in W2) shows ref-intent rate +0.587 at step 30. Trained substrate shows +3.388. **Difference of +2.801, ~6× more pursuit of the prior intention.**

The substrate carries its prior state through the gap in a way that a reset/fresh-instance system cannot. This is the LLM-stateless-instance refutation, in mechanism if not at scale.

## What overall survives

1. **Substrate-identity (substrate ≠ reset/fresh-instance):** SURVIVES STRONGLY. A trained substrate is behaviorally distinct from a reset substrate.
2. **Multi-day identity preservation:** SURVIVES CLEANLY across all components.
3. **Persistence-across-sleep with episodic wipe:** SURVIVES (the substrate retains autobiographical content from its own state without scaffolding).

## What was oversold or fails

1. **Compression efficiency vs trivial baselines:** FAILS in this regime. Trivial counting matches/beats substrate.
2. **FLOPs framing as universal advantage:** PARTIALLY FAILS. Smart-RAG is faster than substrate.
3. **"Conflict-specific stickiness":** OVERSOLD. The behavior is general weight-persistence, not reward-conflict-resistance.

## What this means

The CONCEPTUAL claim — substrate-as-identity vs LLM-stateless-instance — survives. The architecture really does carry its prior self forward through the gap, distinct from how a fresh instance would behave.

The EFFICIENCY claim — "wins on FLOPs/memory" — does not survive in this toy regime. Trivial baselines beat or match the substrate.

This is consistent with the architecture's expected value-add: **substrate-identity systems should win in regimes where Hebbian smoothing, learned representations, and noise tolerance matter — high-dim observations, partial observability, distractors, longer temporal dependencies.** The toy 6-flavor world has none of those properties; it's a regime where explicit counting trivially wins.

## v4 directive

Stop trying to demonstrate efficiency in this regime. Move to a regime where the architecture's claimed advantages should show:

1. **Partial observability** — agent observes flavor only every 3rd step. Explicit counting fails because pair-relations aren't directly visible. Substrate's Hebbian smoothing should bridge gaps.
2. **Distractor noise** — 20% of observations are random. Substrate's replay-averaging should denoise; explicit counting has no smoothing.
3. **Larger state space** — 50+ flavors, 100+ positions, 5+ step temporal dependencies. Explicit counting's memory grows; substrate is bounded.
4. **Tiny transformer baseline** — not just RAG. Match parameter count, compare on accuracy and FLOPs.

If the substrate beats explicit counting in any of these regimes, we have a real efficiency claim. If not, the thesis is "substrate-identity is a different paradigm with the same efficiency as explicit lookup," which is conceptually interesting but not "Better than LLM" except in a narrow architectural sense.

## Honest position

We have a working substrate-identity demonstration. We have NOT demonstrated that this beats the simplest possible baseline at any task we've tested. The thesis is alive but considerably narrower than v3's verdict initially suggested.
