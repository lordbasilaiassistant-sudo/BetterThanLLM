# CLS-style continual learning — state of the art (2026-05)

Scope: built systems with code, not surveys. Target: substrate-drift + replay, not RAG-with-buffer.

## Shipping code worth reading

### 1. CH-HNN (Nature Communications, Feb 2025) — `qqish/CH-HNN`
Hybrid ANN (slow/cortex prior) + SNN (fast/hippocampal) trained on Split-CIFAR-100, Tiny-ImageNet, pMNIST/sMNIST. Task-agnostic, class-incremental, no rehearsal buffer growth. Two anatomically distinct nets connected by feedforward + feedback loops. Closest published architecture to our cortex/hippocampus split. **Honest caveat:** "episode inference modulation," not a true offline replay phase. https://github.com/qqish/CH-HNN · paper https://www.nature.com/articles/s41467-025-56405-9

### 2. Bazhenov lab — sleep replay as offline phase
Sleep-Like Unsupervised Replay (Tadros/Krishnan/Bazhenov, *Nat. Commun.* 2022; AAAI 2024 + 2025 follow-ups, arXiv 2402.10956, 2410.16154). Explicit **off-line phase** with local Hebbian plasticity + noisy input — not a buffer replay. Constrains weights to overlap manifold of old + new tasks. Spiking variant: PLOS Comp. Bio. 2022 (`pcbi.1010628`). This is the only group treating sleep as architectural, not auxiliary. https://www.nature.com/articles/s41467-022-34938-7

### 3. CLS-ER (ICLR 2022) — `NeurAI-Lab/CLS-ER` and DualNet (NeurIPS 2021) — `phquang/DualNet`
Dual short/long semantic memory (CLS-ER) and slow/fast network split (DualNet). Strong on Seq-CIFAR10, S-CIFAR100, S-MNIST. Both still gradient-step-based — closer to "transformer-with-replay-buffer" than substrate change, but the codebases are clean and the dual-timescale split is exactly the wiring we need. Arani et al. https://github.com/NeurAI-Lab/CLS-ER · Pham et al. https://github.com/phquang/DualNet

### 4. Tolman-Eichenbaum Machine — `jbakermans/torch_tem`, `djcrw/generalising-structural-knowledge`
Whittington 2020 (*Cell*). MEC/grid + LEC/place factorization, generalizes structure across environments. TEM-transformer follow-up (Whittington 2022, ICLR) shows transformers with recurrent positional encoding recover place/grid cells. Author's 2024 *Neuron* paper "A Cellular Basis for Mapping Behavioural Structure" extends to non-spatial relational structure. Code is research-grade, low commit activity (~7 commits), but the math is the cleanest published instantiation of structure/content factorization.

### 5. van de Ven brain-inspired replay — `GMvandeVen/brain-inspired-replay`
*Nat. Commun.* 2020. Generative replay of internal representations via context-modulated feedback — no stored data. SOTA on Split-CIFAR-100 class-incremental. Active follow-up "Insights from Brain-Inspired Replay" arXiv 2509.00047 (Sep 2025). Stable, well-documented, the reference implementation everyone benchmarks against.

### 6. NeuralPlayground — `SainsburyWellcomeCentre/NeuralPlayground`
Standardized hippocampus/EC environment harness; bundles TEM. Saves us writing the gridworld substrate.

## Successor representations
`manantomar/DSR`, `mike-gimelfarb/deep-successor-features-for-transfer`, `Hanlard/CPPO` (ICLR 2024). Mostly transfer-learning framing, not CLS. Useful as the "what does the cortex predict" prior for our slow weights — skip for v1.

## Where these still lose to LLMs
None do open-domain language. None scale past CIFAR-100 / Tiny-ImageNet. CH-HNN is class-incremental on ~100 classes; LLMs are zero-shot on millions of concepts. None have a default-mode-process running between inputs.

## Where they win
Catastrophic forgetting on task-shift WITHOUT freezing weights AND without storing raw data: CH-HNN, Bazhenov sleep replay, van de Ven generative replay all clear this. LLMs cannot — they freeze and RAG.

## Build/skip
**Build on:** Bazhenov sleep-phase mechanism + CH-HNN dual-net split + van de Ven generative replay for the consolidation channel + NeuralPlayground for environments.
**Skip:** continual-learning surveys, LLM+RAG+rehearsal-buffer hybrids, CL-RL transfer-only benchmarks.
