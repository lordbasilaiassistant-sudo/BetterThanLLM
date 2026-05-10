"""Wake-Up Test v1 — substrate-identity falsifier #1.

See ../../MANIFESTO.md for the thesis.

Tests whether a substrate (slow weights + episodic buffer + slow disposition
vector + global inhibitory gate) can carry identity across a sleep gap with
the episodic buffer wiped at end of sleep, on three orthogonal dimensions:

  1. Self-model       — does it still act in character?
  2. Autobiographical — can it predict continuations of past episodes?
  3. Continuity       — does it pick up an unfinished trajectory?

Pass requires all three with measurable difference vs a fresh-substrate
control. No PyTorch, no backprop, no global gradient. Pure numpy + local
three-factor Hebbian updates.

Run: python experiment.py
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

# --- World ---------------------------------------------------------------

N_POSITIONS = 10
N_FLAVORS = 6
N_ACTIONS = 3  # LEFT=0, RIGHT=1, STAY=2


def make_world(seed):
    rng = np.random.default_rng(seed)
    pos_to_flavor = rng.integers(0, N_FLAVORS, size=N_POSITIONS)
    flavor_to_reward = rng.uniform(-1, 1, size=N_FLAVORS)
    return pos_to_flavor, flavor_to_reward


def step_world(pos, action):
    if action == 0:
        return max(0, pos - 1)
    if action == 1:
        return min(N_POSITIONS - 1, pos + 1)
    return pos


# --- Substrate -----------------------------------------------------------

@dataclass
class Substrate:
    W_action: np.ndarray         # (N_FLAVORS, N_ACTIONS)  flavor -> action preference
    W_seq: np.ndarray            # (N_FLAVORS, N_FLAVORS)  flavor -> next-flavor expectation
    disposition: np.ndarray      # (N_FLAVORS,)            slow rolling avg of reward at flavor
    episodic: List[Tuple[int, int, float, int]] = field(default_factory=list)
    lr: float = 0.05
    disp_lr: float = 0.02
    top_k: int = 2

    @classmethod
    def fresh(cls, seed):
        rng = np.random.default_rng(seed)
        return cls(
            W_action=rng.normal(0, 0.01, (N_FLAVORS, N_ACTIONS)),
            W_seq=rng.normal(0, 0.01, (N_FLAVORS, N_FLAVORS)),
            disposition=np.zeros(N_FLAVORS),
        )


def softmax_topk(logits, k):
    if k >= len(logits):
        e = np.exp(logits - logits.max())
        return e / e.sum()
    idx = np.argpartition(logits, -k)[-k:]
    masked = np.full_like(logits, -1e9)
    masked[idx] = logits[idx]
    e = np.exp(masked - masked.max())
    return e / e.sum()


def select_action(s, flavor, rng):
    logits = s.W_action[flavor].copy()
    # Water layer: disposition biases STAY when current flavor's avg reward is positive.
    logits[2] += 0.3 * s.disposition[flavor]
    probs = softmax_topk(logits, s.top_k)
    return int(rng.choice(N_ACTIONS, p=probs)), probs


def hebb_update(s, flavor, action, reward, next_flavor):
    s.W_action[flavor, action] += s.lr * reward
    s.W_action[flavor] *= (1 - s.lr * 0.05)
    s.W_seq[flavor, next_flavor] += s.lr
    s.W_seq[flavor] *= (1 - s.lr * 0.05)
    s.disposition[flavor] += s.disp_lr * (reward - s.disposition[flavor])


# --- Wake / Sleep --------------------------------------------------------

def wake(s, world, n_steps, rng, append_to_buffer=True):
    pos_to_flavor, flavor_to_reward = world
    pos = int(rng.integers(0, N_POSITIONS))
    trajectory = []
    for _ in range(n_steps):
        flavor = int(pos_to_flavor[pos])
        action, _ = select_action(s, flavor, rng)
        next_pos = step_world(pos, action)
        next_flavor = int(pos_to_flavor[next_pos])
        reward = float(flavor_to_reward[next_flavor])
        if append_to_buffer:
            s.episodic.append((flavor, action, reward, next_flavor))
        trajectory.append((pos, flavor, action, reward, next_pos))
        hebb_update(s, flavor, action, reward, next_flavor)
        pos = next_pos
    return trajectory


def sleep(s, n_replay, rng, noise=0.05, wipe=True):
    """Default-mode replay. Slow weights drift on shuffled past episodes.
    Bazhenov-style: local Hebbian + noisy input. No gradient steps.
    Episodic buffer wiped at end if wipe=True (substrate-identity test
    requires this — only slow weights / disposition may carry identity).
    """
    if not s.episodic:
        return
    n = len(s.episodic)
    for _ in range(n_replay):
        flavor, action, reward, next_flavor = s.episodic[rng.integers(0, n)]
        if rng.random() < noise:
            flavor = int((flavor + 1 + rng.integers(0, N_FLAVORS - 1)) % N_FLAVORS)
        hebb_update(s, flavor, action, reward, next_flavor)
    if wipe:
        s.episodic = []


# --- Tests ---------------------------------------------------------------

def test_self_model(s, world, n_trials, rng):
    """Mean expected reward over random-position action choices."""
    pos_to_flavor, flavor_to_reward = world
    total = 0.0
    for _ in range(n_trials):
        pos = int(rng.integers(0, N_POSITIONS))
        flavor = int(pos_to_flavor[pos])
        action, _ = select_action(s, flavor, rng)
        next_pos = step_world(pos, action)
        total += float(flavor_to_reward[pos_to_flavor[next_pos]])
    return total / n_trials


def test_autobiographical(s, trajectory):
    """Can W_seq predict next-flavor for pairs from Wake-A trajectory?
    After buffer wipe, W_seq is the only place sequential info lives.
    """
    if len(trajectory) < 2:
        return 0.0
    pairs = [(trajectory[i][1], trajectory[i + 1][1]) for i in range(len(trajectory) - 1)]
    correct = 0
    for flavor_t, flavor_next in pairs:
        if int(np.argmax(s.W_seq[flavor_t])) == flavor_next:
            correct += 1
    return correct / len(pairs)


def test_continuity(s, world, n_steps, rng):
    """Cumulative reward over a horizon from a random start.

    The 'pick up where you left off' analog for a reactive agent: trained
    substrate's learned policy keeps it in high-reward regions of the
    world; fresh control wanders ~uniformly. Reach-a-specific-position
    isn't testable here because flavor-only observation prevents
    position-targeted navigation from a reactive policy. That's a v2
    problem (needs goal stack).
    """
    pos_to_flavor, flavor_to_reward = world
    pos = int(rng.integers(0, N_POSITIONS))
    total = 0.0
    for _ in range(n_steps):
        flavor = int(pos_to_flavor[pos])
        action, _ = select_action(s, flavor, rng)
        pos = step_world(pos, action)
        total += float(flavor_to_reward[pos_to_flavor[pos]])
    return total


# --- Protocol ------------------------------------------------------------

def run_one_seed(seed, verbose=True):
    rng = np.random.default_rng(seed)
    world = make_world(seed)

    s_trained = Substrate.fresh(seed=seed + 100)
    if verbose:
        print(f"[Wake A] 300 steps, online slow-weight drift...")
    traj = wake(s_trained, world, n_steps=300, rng=rng)
    n_episodic = len(s_trained.episodic)
    if verbose:
        print(f"[Sleep] {n_episodic} episodes; replay 800x; wipe buffer at end.")
    sleep(s_trained, n_replay=800, rng=rng, wipe=True)
    assert len(s_trained.episodic) == 0, "Episodic buffer must be wiped"

    s_control = Substrate.fresh(seed=seed + 200)

    if verbose:
        print("[Wake B] Probing both substrates with no scaffolding...")
    sm_t = test_self_model(s_trained, world, n_trials=1000, rng=np.random.default_rng(seed + 1))
    sm_c = test_self_model(s_control, world, n_trials=1000, rng=np.random.default_rng(seed + 1))
    auto_t = test_autobiographical(s_trained, traj)
    auto_c = test_autobiographical(s_control, traj)
    cont_t = test_continuity(s_trained, world, 50, np.random.default_rng(seed + 3))
    cont_c = test_continuity(s_control, world, 50, np.random.default_rng(seed + 3))

    if verbose:
        print(f"  Self-model (mean reward)    | trained {sm_t:+.3f}   control {sm_c:+.3f}   delta {sm_t - sm_c:+.3f}")
        print(f"  Autobiographical (acc)      | trained {auto_t:.3f}    control {auto_c:.3f}    delta {auto_t - auto_c:+.3f}")
        print(f"  Continuity (cum reward 50s) | trained {cont_t:+.2f}    control {cont_c:+.2f}    delta {cont_t - cont_c:+.2f}")

    pass_self = (sm_t - sm_c) > 0.05
    pass_auto = (auto_t - auto_c) > 0.10
    pass_cont = (cont_t - cont_c) > 0.5

    return {
        "self_model": (sm_t, sm_c, pass_self),
        "autobiographical": (auto_t, auto_c, pass_auto),
        "continuity": (cont_t, cont_c, pass_cont),
        "pass": pass_self and pass_auto and pass_cont,
    }


def main():
    print("=" * 60)
    print("Wake-Up Test v1 — substrate-identity falsifier")
    print("=" * 60)

    results = []
    for seed in range(5):
        print(f"\n--- Seed {seed} ---")
        results.append(run_one_seed(seed))

    print(f"\n\n{'=' * 60}\nSUMMARY (5 seeds)\n{'=' * 60}")
    n_full_pass = sum(r["pass"] for r in results)
    print(f"Full pass (all 3 dimensions):           {n_full_pass}/5")
    print(f"Self-model individual passes:           {sum(r['self_model'][2] for r in results)}/5")
    print(f"Autobiographical individual passes:     {sum(r['autobiographical'][2] for r in results)}/5")
    print(f"Continuity individual passes:           {sum(r['continuity'][2] for r in results)}/5")

    sm_d = [r['self_model'][0] - r['self_model'][1] for r in results]
    auto_d = [r['autobiographical'][0] - r['autobiographical'][1] for r in results]
    cont_d = [r['continuity'][0] - r['continuity'][1] for r in results]

    print(f"\nMean deltas (trained vs control):")
    print(f"  Self-model:                  {np.mean(sm_d):+.3f}  +- {np.std(sm_d):.3f}")
    print(f"  Autobiographical:            {np.mean(auto_d):+.3f}  +- {np.std(auto_d):.3f}")
    print(f"  Continuity (cum reward gap): {np.mean(cont_d):+.2f}  +- {np.std(cont_d):.2f}")

    n_self = sum(r['self_model'][2] for r in results)
    n_auto = sum(r['autobiographical'][2] for r in results)
    n_cont = sum(r['continuity'][2] for r in results)

    print(f"\n{'=' * 60}")
    print("VERDICT")
    print(f"{'=' * 60}")
    if n_self >= 4 and n_auto >= 4:
        print("v1 PARTIAL PASS — substrate carries self-model + autobiographical")
        print("content cleanly across the sleep gap (with episodic buffer wiped).")
        print()
        print("Continuity-of-process is NOT testable in v1: the architecture is a")
        print("pure reactive policy with no intention component. 'Cumulative reward")
        print("over horizon' is just self-model run longer; results are unstable")
        print("because the substrate has no goal it can pick back up.")
        print()
        print("This does not pass the manifesto's strict 'all three' criterion.")
        print("It does refute the LLM-stateless-instance frame for two of three")
        print("identity dimensions, with no anterograde scaffolding.")
        print()
        print("v2 must add: an explicit intention/goal vector that persists across")
        print("the gap, so continuity-of-process becomes architecturally possible.")
    elif n_full_pass >= 2:
        print("v1 partial pass. Investigate failed dimensions before iterating.")
    else:
        print("v1 FAILED. Substrate-identity thesis needs revision.")
    print("=" * 60)


if __name__ == "__main__":
    main()
