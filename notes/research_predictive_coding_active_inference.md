# PCN + Active Inference: Shipping Code vs. Talk (2023-2026)

## Predictive Coding Networks - actual code

- **PCX (Pinchetti, Salvatori, Bogacz et al., "Benchmarking Predictive Coding Networks -- Made Simple," arXiv:2407.01163, 2024)** - JAX library, PyTorch-like API, JIT-compiled. Repo: https://github.com/liukidar/pcx. Reaches backprop-parity on **CIFAR-10/100, Tiny-ImageNet on conv nets up to 5-7 layers**. ImageNet results "comparable to backprop" but only at modest depth. This is the de-facto reference implementation.
- **JPC (Buckley lab, Sussex)** - https://github.com/thebuckleylab/jpc. Flexible JAX PCN inference, more research-y, lighter than PCX.
- **muPC (Innocenti et al., arXiv:2505.13124, NeurIPS 2025)** - Depth-muP parameterization that trains stable **128-layer residual PCNs**. First credible answer to the depth-collapse problem. Code released.
- **Predictive Coding Light (Nature Comms, 2025)** - hierarchical spiking PCN, suppresses predictable spikes, transmits compressed deltas. Closest thing to genuine "compute-on-surprise" actually measured on hardware.
- **Millidge "Introduction to PCNs for ML" (arXiv:2506.06332, 2025)** - the current canonical tutorial; he himself has moved to Mamba/Zamba2 for production work, which is a tell.

## Active Inference - actual code

- **pymdp** (https://github.com/infer-actively/pymdp) - discrete-state POMDPs only; mature, used in 90% of academic AIF papers; toy-scale.
- **RxInfer.jl** (https://github.com/reactivebayes/RxInfer.jl) - reactive message passing on factor graphs, fastest AIF runtime; RxInferServer + SDKs landed Q1 2025.
- **ActiveInference.jl** (MDPI Entropy 27/1/62, 2025) - Julia port of pymdp.
- **AXIOM (VERSES, arXiv:2505.24784, May 2025)** - https://github.com/VersesTech/axiom. Object-centric mixture models + active inference, **no NN, no backprop, no replay**. Beats DreamerV3 + BBF on Gameworld-10k with ~10k frames. The single most credible "non-LLM beats SOTA" result in the space.
- **Deep Active Inference for Delayed/Long-Horizon Envs** (arXiv:2505.19867, 2025) - hybrid deep + AIF agents.

## Honest weaknesses

1. **PCN sparse-compute is theoretical, not measured.** No PCN paper reports FLOPs/Joule head-to-head with a transformer. PCL is the only one with energy numbers, and only because it runs on neuromorphic spiking hardware - not the same architecture.
2. **PCN depth ceiling is real.** Standard PC degrades past 5-7 layers (vanishing-error problem); muPC fixes ResNets specifically, not Transformers/general graphs.
3. **AIF has a state-space ceiling.** pymdp/RxInfer are Bayesian-exact - state explosion is brutal. AXIOM only works because object-centric priors make state finite.
4. **No production AIF agent exists outside VERSES demos and a handful of robotics labs.** "Deployed in autonomous driving / clinical" claims don't survive grep.
5. **Genius/AXIOM is closed-business + open-research** - VERSES is a public co. with skeptics; treat marketing claims (>90% data reduction, "beats Google") as best-case.

## Worth forking

1. **VersesTech/axiom** - working non-NN agent, MIT-ish license, Atari/Gameworld benchmarks reproducible.
2. **liukidar/pcx** - cleanest PCN substrate; build the BetterThanLLM perception stack here.
3. **reactivebayes/RxInfer.jl** - if we go reactive/factor-graph route for the action loop.
