# Prior Art: Always-On Persistent-Substrate Agents

Date: 2026-05-09
Verdict: The substrate-is-identity / always-on idle-loop space is **largely unclaimed**. Adjacent work exists in fragments; no one has integrated them into a working system where the substrate itself runs continuously and identity is bound to that process.

## What exists (adjacent, not the thing)

**1. Sleep-time / dreaming compute (LLM scaffolding, not substrate)**
- Letta "sleeptime agents" (2024-2025): background LLM call edits memory blocks every N turns. Memory-as-text mutates; weights do not.
- Anthropic "Dreaming" for Managed Agents (2026): replays prior action sequences, tunes prompts/memory. Same pattern.
- Google "Always-On Memory Agent" (open-sourced 2025): 30-min cron consolidates SQLite memories via LLM. Cron job, not a process.
- All of these are call-response systems with a second call-response loop on a timer. Substrate is frozen.

**2. Test-Time Training / In-Place TTT (substrate mutates, but only on input)**
- TTT-E2E (arxiv 2512.23675), In-Place TTT (arxiv 2604.06169): subset of MLP weights ("fast weights") update via next-token loss during inference.
- Closest hit on substrate mutation post-deployment. But: updates are input-driven, no idle loop, no replay-when-quiet, mutations typically scoped to a context window then discarded.

**3. Fast Weights lineage**
- Ba/Hinton/Mnih 2016 (arxiv 1610.06258): two-timescale weights, slow + fast, fast decay between steps. Toy-scale, never deployed as an always-on agent.
- Miconi/Clune/Stanley "Differentiable Plasticity" (1804.02464) and "Backpropamine" (neuromodulated plasticity): plastic connections that update online via Hebbian rules. Research artifacts, not deployed agents.

**4. World models with imagination (Dreamer, etc.)**
- Hafner Dreamer v3 (Nature 2025): rollouts in a learned model. The "dreaming" happens during training to generate gradient signal, not while idle in deployment.

**5. DMN-inspired AI**
- ResearchGate / Medium thinkpieces (Miriam W. 2024, Tutuncuoglu "NeuroDream" SSRN 2025) propose DMN-style background processing. NeuroDream shows latent replay reducing forgetting. All offline-replay-during-training, none always-on-in-deployment.
- "Dark control: DMN as RL agent" (Wiley 2020) is interpretive neuroscience, not an architecture.

**6. Reservoir computing / neuromodulation / brain organoids**
- Continuous-time substrates exist (literally always-on physically). But identity/learning lives in a downstream readout layer trained offline; the reservoir itself is fixed-random.

## The unclaimed ground

Nothing in the literature combines all four:
1. Substrate (weights/state) that mutates online, post-deployment, persistently.
2. An idle loop that runs *when no input arrives* doing replay/consolidation/self-modeling.
3. Identity bound to the running process — kill the process, lose the agent.
4. Non-LLM (or at least not just-an-LLM-with-memory).

Closest convergence: TTT + plastic networks + sleeptime agents — but no one has glued them. The "substrate IS identity" framing is, as far as this search reaches, original.

## Honest caveat

Cognitive-architecture lineage (Soar, ACT-R, OpenCog) has always-running daemons but with hand-coded symbolic substrates, not learned ones. Worth a deeper look if claiming full novelty.
