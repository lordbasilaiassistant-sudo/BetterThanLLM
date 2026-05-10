# Mortal Computation — Research Note

**Date:** 2026-05-09
**Question:** Hinton's mortal-computation thesis. Who else thinks substrate IS identity, and who has actually *built* anything?

## The canonical sources

- **Hinton, "The Forward-Forward Algorithm: Some Preliminary Investigations"** — arXiv:2212.13345, Dec 2022. Last section ("Mortal Computation") is the load-bearing one for us. Argument: drop hardware/software separation, let weights be inseparable from analog substrate, learn in-place with FF (no backprop needed). FF itself is a layer-local contrastive rule. Code: many community ports, Hinton's own is a toy MNIST.
- **Hinton, Romanes Lecture, Oxford, 19 Feb 2024** — "Will digital intelligence replace biological intelligence?" Reframes mortal computation as the *cost* of analog efficiency: lower energy, but algorithm dies with hardware, and no parallel-copy weight-sharing. Concludes digital wins on capability; analog wins on efficiency. **Important for us:** Hinton himself frames mortal-computation as a hardware story, not an identity story. We're using his term but pushing further.
- **Self-Assembling Brain (Hiesinger) commentary, 2023** — connects mortal computation to developmental biology. Theory only.

## Follow-ups worth our time

- **Ororbia & Friston, "Mortal Computation: A Foundation for Biomimetic Intelligence"** — arXiv:2311.09589, Nov 2023 (rev. Feb 2024). Frames mortal computation through Markov blankets + free-energy principle. **Theory paper.** No implementation in this work, but Ororbia's lab (NACLab) ships **ngc-learn** (github.com/NACLab/ngc-learn), JAX-based predictive-coding / SNN library, BSD-3. This is the closest thing to a runnable substrate-style codebase we found. Worth forking for predictive-coding primitives.
- **Kleiner, "Consciousness qua Mortal Computation"** — arXiv:2403.03925, Mar 2024. Pure philosophy. Skip.
- **Salvatori et al., "A Stable, Fast, and Fully Automatic Learning Algorithm for Predictive Coding Networks"** — arXiv:2212.00720. Incremental PC (iPC), continuous-time, no alternating phases. Real algorithmic progress.
- **PCN-TA, arXiv:2510.25993** (IROS 2025) — predictive coding with temporal amortization, online learning, ~10% fewer weight updates than backprop. Online-learning evidence.
- **FF empirical state, 2024-2025:** Self-Contrastive FF (Nature Comms 2025), CNN-FF (Sci Reports 2025), CIFAR-10 ~84.7% with random-direct-feedback. **Still well below backprop, still struggles past CIFAR-10, no LLM-scale.** Forward-only learning is alive but losing on capability.

## Critiques

- **Inference.vc (Ferenc Huszár), "Mortal Komputation"** — argues Hinton underestimates one-shot human learning; FF is essentially noisy evolution-strategies and "doesn't scale to any decent sized learning problem."
- **Bengio:** no specific published critique of mortal computation found. He works adjacent (equilibrium propagation, GFlowNets) but doesn't engage Hinton's framing directly.
- **Schmidhuber:** priority disputes with Hinton generally; nothing specific on mortal computation.

## Has anyone *shipped* a substrate-bound system?

**No.** Closest:
- IBM **aihwkit** — analog in-memory training simulator, weights still digital-portable in practice.
- BrainScaleS-2, Loihi 2, DYNAP-SE — analog/mixed-signal neuromorphic chips with on-chip plasticity, mostly running SNNs at MNIST/EuroSAT scale. Substrate-bound in principle, but none ship a default-mode loop, episodic store, or substrate-as-identity claim.
- ngc-learn — software simulator, not substrate-bound, but the cleanest predictive-coding/biomimetic toolkit.

**Vapor vs. shipping line:** Hinton, Ororbia, Kleiner, Kleiner-style follow-ups = talk. ngc-learn, Salvatori's iPC, FF-CNN papers, BrainScaleS demos = shipping but small. No one has built what the manifesto asks for.
