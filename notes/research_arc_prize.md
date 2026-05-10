# ARC Prize 2024 / 2025 — What Actually Won

Date: 2026-05-09. Source-cited. Compute numbers where public.

## Kaggle Compute Constraint (both years)
Single Kaggle VM, ~12 hr runtime, ~4 small GPUs (P100-class in 2024; ~$0.20/task budget in 2025), no internet, must open-source to win prize. This is the "small compute" lane. [arcprize/2024], [arcprize/2025].

## ARC Prize 2024 (ARC-AGI-1, private eval)
- **MindsAI: 55.5%** — top score, withheld solution, prize-ineligible.
- **ARChitects (winner): 53.5%** — test-time training on a small LLM (LLaMA-class). Won by open-sourcing. [arcprize/2024-tech-report, arXiv 2412.04604].
- Frontier-LLM context: o3 hit ~76–88% on ARC-AGI-1 *high-compute* (~$20–$3,000/task), proving LLMs catch up only by burning compute. [arcprize/o3-breakthrough].

## ARC Prize 2025 (ARC-AGI-2, private eval, ran Mar–Nov 2025)
1. **NVARC — 24.03%** (NVIDIA Kaggle Grandmasters). Fine-tuned **Qwen2-VL 4B**, tokenizer cut to 16 tokens, single transformer block iterated TRM-style, synthetic data + TTT. ~$0.20/task. [developer.nvidia.com/blog, github.com/1ytic/NVARC].
2. **ARChitects — 16.53%** — 2D-aware **masked-diffusion LM** with recursive self-refinement.
3. **MindsAI — 12.64%** — TTFT + augmentation ensembles + tokenizer dropout.
[arcprize.org/blog/arc-prize-2025-results-analysis]

Grand Prize (85%) **unclaimed**. 1,455 teams, 15,154 entries.

## Paper Awards — the genuinely small / non-LLM stuff
- **TRM (Jolicoeur-Martineau, Samsung SAIL Montréal, 1st paper)**: **7M params**, 2-layer recursive net. **45% ARC-AGI-1, 7.8% ARC-AGI-2** — beats Gemini 2.5 Pro (4.9%) and o3-mini-high (3.0%) on AGI-2 at a tiny fraction of compute. arXiv 2510.04871. NOT pure program synthesis — it's iterative latent refinement, sub-symbolic.
- **SOAR (Pourcel/Colas/Oudeyer, 2nd paper)**: LLM-driven **evolutionary program synthesis** + hindsight self-fine-tuning. **52% ARC-AGI-1 public test, 42.75% with 14B model**, beats GPT-4.1 one-shot and o3-mini (33%). arXiv 2507.14172.
- **CompressARC (3rd paper)**: **76K params**, trained *only* at test time, no pretraining at all.

## Frontier LLM scores on ARC-AGI-2 (high compute)
- GPT-5.2: 52.9% (~$30/task class), GPT-5.5: ~85%, Gemini 3.1 Pro: 77.1%, Claude Opus 4.5: 37.6%, Confluence Lab refinement on Gemini 3 Pro: 97.9% Apr 2026. [llm-stats.com/arc-agi-v2, sanj.dev/arcprize-leaderboard].
- Best efficiency: Berman 2025 ~80% at **$8.42/task** (LLM + program-search hybrid).

## Honest verdict
**Pure compositional/program-synthesis is NOT decisively winning.** Every Kaggle 2025 podium entry is a small-LLM-with-test-time-training (TTT). The ARChitects switched FROM autoregressive TTT TO masked-diffusion. SOAR (closest to DreamCoder lineage) won a *paper* award, not the score race.

**But at small compute, neural-symbolic hybrids dominate.** TRM at 7M params humiliates frontier LLMs on ARC-AGI-2 per FLOP. NVARC at $0.20/task ≈ 100× cheaper than mid-tier frontier setups.

**Chollet's 2025/26 line**: hybrid of discrete program search + LLM-guided search wins. TTT becomes mainstream from 2026. Pure LLM scaling is *not* the path; pure program synthesis isn't either. [arcprize.org/blog/beat-arc-agi-deep-learning-and-program-synthesis, dwarkesh.com/p/francois-chollet].

## Implication for BetterThanLLM
The winning small-compute paradigm is **tiny iterative/recursive nets + TTT + synthetic data**, not DreamCoder-style symbolic search. SOAR shows program synthesis still has signal but needs an LLM in the loop. A non-LLM bet should look more like TRM (recursive latent refinement) than classical inductive program synthesis.
