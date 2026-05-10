# BetterThanLLM

**A non-LLM AI architecture whose substrate remembers who it is when it wakes up.**

Pure numpy. No backprop. No transformer. Local Hebbian learning + sleep replay + persistent slow weights. Ten identity tests, all passing in toy worlds. Empirical demonstration that an AI can carry individual identity across sleep gaps with the episodic buffer wiped — a property no LLM-stateless-instance system has.

This is research code. Toy-world scope (up to 30 positions × 12 flavors). The architecture does NOT match LLM capability and does NOT beat trivial counting baselines on raw predictive accuracy. What it provides is a different category of capability: substrate-bound continuous identity.

---

## What this demonstrates

Ten independent tests, all passing, code in `experiments/identity_tests/`:

| Test | Result | What it proves |
|---|---|---|
| Single-sleep behavioral continuity | **0.81** cosine | Same person, bounded overnight drift |
| 10-cycle multi-sleep identity | **0.79** to cycle-1 | Identity holds across many sleeps |
| Self vs other (different worlds) | **8/10** strict, +0.22 gap | Distinguishable from peers when experiences differ |
| Episode-specific autobiographical recall | **19/20** pairs | Substrate knows its own past, not generic stats |
| Counterfactual fork (twins separated) | shared-past +0.22, divergence +0.11 | Forks remember shared origin AND grow distinct |
| Identity transfer (deep copy) | **1.0000** | Identity is in the weights — exact copy = same individual |
| 30% adversarial damage | **0.99** retention | Graceful degradation, not catastrophic |
| Cross-world identity preservation | **+0.18** signal | Substrate is more itself than its environment |
| Component ablation | W_action critical, disposition not | Behavior fingerprint = action preferences |
| 50-cycle long-horizon | **0.82** mean to baseline | Identity bounded across very long horizons |

All numbers reproducible with `py experiments/identity_tests/experiment_v4.py && py experiments/identity_tests/experiment_v5.py`.

---

## What this is NOT

- **Not consciousness.** No qualia, no reportability, no self-modeling. We measure behavior, not phenomenology.
- **Not LLM-level capability.** Toy gridworld, not language. The architecture would not write code or hold a conversation.
- **Not an efficiency claim.** A 5-line `counts[f,n] += 1` table beats us on predictive accuracy in toy regimes. See [`experiments/wake_up_test_v3/FACT_CHECK_REPORT.md`](experiments/wake_up_test_v3/FACT_CHECK_REPORT.md) for the audit that pruned bad claims.
- **Not at scale.** Largest demonstration: 30 positions, 12 flavors. Whether the property survives at scale is genuinely unknown.

---

## Why it might matter

The architecture provides a primitive that LLM-based systems structurally cannot: **a persistent individual identity that survives input gaps**. Concrete potential applications, none yet built:

- **Identity layer for LLM agents.** Wrap an LLM session with a substrate carrying personality, dispositions, and autobiographical traces. LLM does language; substrate does *being-someone*. Cross-world identity (T8) suggests this would survive context resets that plain LLM agents can't.
- **Multi-agent discrimination by behavioral signature** instead of tags or IDs.
- **Continual fine-tuning with graceful degradation** (T7 robustness).
- **Long-running coherent agents** for week/month deployments where the agent must "be the same individual" (T10 long-horizon).

These are conjectures, not built products. The repo demonstrates the *primitive*; productization is open work.

---

## Architecture in 30 seconds

```
Substrate components (pure numpy):
- W_action      : flavor → action preference         (reactive policy)
- W_trans       : flavor × action → next-flavor      (learned transitions)
- intention     : per-flavor goal preference         (persistent goals)
- disposition   : per-flavor rolling reward avg      ("water layer")
- episodic      : ring buffer (WIPED at end of sleep — no memory file)

Wake:  Hebbian online updates from world interaction
Sleep: shuffled replay + 5% noise; buffer wiped at end
Action: reactive (W_action + disposition) + planning (W_trans @ intention) + top-K gate
```

Cardinal rule: **No anterograde scaffolding.** After sleep, the substrate gets no system prompt, no memory file, no curated context. Whatever it retained is all it has.

Full design rationale in [`MANIFESTO.md`](MANIFESTO.md).

---

## Quick start

```bash
git clone https://github.com/<your-handle>/BetterThanLLM
cd BetterThanLLM
py experiments/identity_tests/experiment_v4.py    # tests T1-T7, ~30s
py experiments/identity_tests/experiment_v5.py    # tests T8-T10, ~30s
```

Requires Python 3.10+ and numpy. No PyTorch. No GPU. Deterministic with fixed seeds.

---

## Repo layout

```
BetterThanLLM/
├── README.md                    # this file
├── FINDINGS.md                  # comprehensive results summary (read this next)
├── MANIFESTO.md                 # full thesis + decision rules + audit history
├── results.json                 # structured numeric results
├── LICENSE                      # MIT
├── STATUS.md                    # "research code, no support"
├── log/
│   ├── 2026-05-09.md            # day 1 narrative
│   └── 2026-05-10.md            # day 2 narrative (audit + identity battery)
├── notes/                       # prior-art research notes
│   ├── research_arc_prize.md
│   ├── research_cls_continual_learning.md
│   ├── research_mortal_computation.md
│   ├── research_persistent_agents.md
│   └── research_predictive_coding_active_inference.md
└── experiments/
    ├── wake_up_test_v1/          # initial 3-reconstitution test
    ├── wake_up_test_v2/          # added intention vector + W_trans
    ├── wake_up_test_v3/          # conflict, multi-day, RAG baseline
    │   └── FACT_CHECK_REPORT.md  # the audit that pruned overclaims
    ├── wake_up_test_v4/          # trade-off characterization
    └── identity_tests/           # final 10-test battery (v1-v5)
```

`FINDINGS.md` is the citable summary. `MANIFESTO.md` is the project's design document. `experiments/` is reproducible code.

---

## Theoretical context

This work sits in the lineage of:

- **Mortal computation** (Hinton, 2022) — substrate-bound rather than weight-portable AI. We push his framing further: from hardware-efficiency claim to identity claim.
- **Complementary learning systems** (McClelland/McNaughton/O'Reilly, 1995) — fast hippocampal episodic + slow cortical semantic, tied by replay. Directly instantiated here.
- **Sleep replay as architectural phase** (Bazhenov lab, 2022-2025) — sleep as a separate computational mode with local Hebbian rules, not a training trick.
- **Predictive coding networks** (Rao-Ballard 1999, Ororbia/Friston/Salvatori 2023+) — top-down prediction, bottom-up error. Influences the slow-weight design.
- **Active inference** (Friston) — actions chosen to minimize expected free energy. Closest existing system: AXIOM (VERSES, 2025).

Detailed prior-art notes in [`notes/`](notes/).

---

## License

MIT. See [`LICENSE`](LICENSE).

---

## Status

Research code. **Not actively maintained for outside use.** No support, no contributions accepted at this time. See [`STATUS.md`](STATUS.md) for details.

If you find this useful and want to build on it, fork freely. Don't expect responses to issues or PRs.

---

## Topics

`non-llm-ai` `substrate-identity` `continual-learning` `hebbian-learning` `mortal-computation` `sleep-replay` `complementary-learning-systems` `numpy` `research-code` `agent-identity` `behavioral-signature` `consciousness-research`

---

## $MNEME (token)

A token tied to this repo exists on Base. **It is a memetic / discoverability artifact, not a product, not investment advice, and the architecture is not integrated with it in any way.** The repo is the research; the token is just a marker. Mneme — Greek goddess of memory, original Muse — fits the "remembers who it is" thesis.

| Field | Value |
|---|---|
| Network | Base (chainId 8453) |
| Contract | [`0xb762138166ca5dcfa3c8c7d4b9c8616a790dab07`](https://basescan.org/token/0xb762138166ca5dcfa3c8c7d4b9c8616a790dab07) |
| Symbol | MNEME |
| Total supply | 100,000,000,000 |
| Pair | WETH (Uniswap V4 via Clanker) |
| Fee | 1% static + 15s anti-snipe |
| Deployer | THRYX `0x7a3E…12E334` |
| Deploy tx | [`0x71108c08…dbfaba1`](https://basescan.org/tx/0x71108c0889c4aa511a0d7c5546cfa098c8772b4bbdc5034f560402b38dbfaba1) |

**Live chart:** [Dexscreener · MNEME / WETH](https://dexscreener.com/base/0xb762138166ca5dcfa3c8c7d4b9c8616a790dab07) · [Clanker page](https://www.clanker.world/clanker/0xb762138166ca5dcfa3c8c7d4b9c8616a790dab07)

[![MNEME chart](https://io.dexscreener.com/dex/chart-images/v2/base/0xb762138166ca5dcfa3c8c7d4b9c8616a790dab07.png?theme=dark)](https://dexscreener.com/base/0xb762138166ca5dcfa3c8c7d4b9c8616a790dab07)

(Chart image is fetched live from Dexscreener; will populate after sufficient on-chain trades.)
