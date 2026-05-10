"""Identity Tests v3 — bigger world, no personality bias.

v2 with personality bias hurt Test 3 (narrowed each substrate's experience).
The cleanest path to higher episode-specific recall: bigger world giving
more room for naturally divergent trajectories.

This file IS self-contained and copies the substrate code with parameterized
world size. Direct-comparison v3 to v1 should make the world-size effect
visible.

Run: py experiment_v3.py
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

# --- Parameterized world ------------------------------------------------

N_POSITIONS = 20    # was 10 in v1
N_FLAVORS = 10      # was 6 in v1
N_ACTIONS = 3


def make_world(seed):
    rng = np.random.default_rng(seed)
    pos_to_flavor = rng.integers(0, N_FLAVORS, size=N_POSITIONS)
    flavor_to_reward = rng.uniform(-1, 1, size=N_FLAVORS)
    if flavor_to_reward.max() < 0.5:
        flavor_to_reward[int(np.argmax(flavor_to_reward))] = float(rng.uniform(0.5, 1.0))
    return pos_to_flavor, flavor_to_reward


def step_world(pos, action):
    if action == 0:
        return max(0, pos - 1)
    if action == 1:
        return min(N_POSITIONS - 1, pos + 1)
    return pos


# --- Substrate ----------------------------------------------------------

@dataclass
class Substrate:
    W_action: np.ndarray
    W_trans: np.ndarray
    disposition: np.ndarray
    intention: np.ndarray
    episodic: List[Tuple[int, int, float, int]] = field(default_factory=list)
    lr: float = 0.05
    disp_lr: float = 0.02
    intent_lr: float = 0.04
    plan_weight: float = 0.6
    top_k: int = 2

    @classmethod
    def fresh(cls, seed):
        rng = np.random.default_rng(seed)
        return cls(
            W_action=rng.normal(0, 0.01, (N_FLAVORS, N_ACTIONS)),
            W_trans=rng.normal(0, 0.01, (N_FLAVORS, N_ACTIONS, N_FLAVORS)),
            disposition=np.zeros(N_FLAVORS),
            intention=np.zeros(N_FLAVORS),
        )


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def softmax_topk(logits, k):
    if k >= len(logits):
        return softmax(logits)
    idx = np.argpartition(logits, -k)[-k:]
    masked = np.full_like(logits, -1e9)
    masked[idx] = logits[idx]
    return softmax(masked)


def select_action(s, flavor, rng):
    react_logits = s.W_action[flavor].copy()
    react_logits[2] += 0.3 * s.disposition[flavor]
    plan_logits = np.zeros(N_ACTIONS)
    for a in range(N_ACTIONS):
        next_dist = softmax(s.W_trans[flavor, a])
        plan_logits[a] = next_dist @ s.intention
    logits = react_logits + s.plan_weight * plan_logits
    probs = softmax_topk(logits, s.top_k)
    return int(rng.choice(N_ACTIONS, p=probs)), probs


def hebb_update(s, flavor, action, reward, next_flavor):
    s.W_action[flavor, action] += s.lr * reward
    s.W_action[flavor] *= (1 - s.lr * 0.05)
    s.W_trans[flavor, action, next_flavor] += s.lr
    s.W_trans[flavor, action] *= (1 - s.lr * 0.05)
    s.disposition[flavor] += s.disp_lr * (reward - s.disposition[flavor])
    baseline = float(s.disposition.mean())
    relative = reward - baseline
    if relative > 0.05:
        s.intention[next_flavor] += s.intent_lr * relative
        s.intention *= (1 - s.intent_lr * 0.05)


def wake(s, world, n_steps, rng):
    pos_to_flavor, flavor_to_reward = world
    pos = int(rng.integers(0, N_POSITIONS))
    trajectory = []
    for _ in range(n_steps):
        flavor = int(pos_to_flavor[pos])
        action, _ = select_action(s, flavor, rng)
        next_pos = step_world(pos, action)
        next_flavor = int(pos_to_flavor[next_pos])
        reward = float(flavor_to_reward[next_flavor])
        s.episodic.append((flavor, action, reward, next_flavor))
        trajectory.append((pos, flavor, action, reward, next_pos))
        hebb_update(s, flavor, action, reward, next_flavor)
        pos = next_pos
    return trajectory


def sleep(s, n_replay, rng, noise=0.05, wipe=True):
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


# --- Identity tests -----------------------------------------------------

def behavioral_signature(s, n_samples=200, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    sig = np.zeros((N_FLAVORS, N_ACTIONS))
    counts = np.zeros(N_FLAVORS)
    for _ in range(n_samples):
        flavor = int(rng.integers(0, N_FLAVORS))
        for _ in range(20):
            action, probs = select_action(s, flavor, rng)
            sig[flavor] += probs
            counts[flavor] += 1
    counts = np.maximum(counts, 1)
    sig /= counts[:, None]
    return sig


def signature_similarity(sig_a, sig_b):
    a = sig_a.flatten()
    b = sig_b.flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def behavioral_continuity(seed, n_wake=500, n_replay=1000):
    rng = np.random.default_rng(seed)
    world = make_world(seed)
    s = Substrate.fresh(seed=seed + 100)
    wake(s, world, n_wake, rng)
    sig_pre = behavioral_signature(s, rng=np.random.default_rng(seed + 555))
    sleep(s, n_replay, rng, wipe=True)
    sig_post = behavioral_signature(s, rng=np.random.default_rng(seed + 555))
    return signature_similarity(sig_pre, sig_post)


def self_vs_other(n_substrates=10, n_wake=500, n_replay=1000, same_world=True):
    pre_sigs, post_sigs = [], []
    for i in range(n_substrates):
        world_seed = 42 if same_world else (42 + i * 1009)
        world = make_world(world_seed)
        rng = np.random.default_rng(world_seed * 1000 + i * 7919)
        s = Substrate.fresh(seed=i * 31337)
        wake(s, world, n_wake, rng)
        pre_sigs.append(behavioral_signature(s, rng=np.random.default_rng(i + 1234)))
        sleep(s, n_replay, rng, wipe=True)
        post_sigs.append(behavioral_signature(s, rng=np.random.default_rng(i + 1234)))
    sim = np.zeros((n_substrates, n_substrates))
    for i in range(n_substrates):
        for j in range(n_substrates):
            sim[i, j] = signature_similarity(post_sigs[i], pre_sigs[j])
    correct = sum(int(np.argmax(sim[i])) == i for i in range(n_substrates))
    self_sims = np.diag(sim)
    mask = ~np.eye(n_substrates, dtype=bool)
    other_sims = sim[mask]
    return {
        "correct": correct, "n": n_substrates,
        "mean_self": float(self_sims.mean()),
        "mean_other": float(other_sims.mean()),
        "gap": float(self_sims.mean() - other_sims.mean()),
    }


def episode_specific_recall(seed, n_wake=500, n_replay=1000):
    world = make_world(seed)
    rng_a = np.random.default_rng(seed + 1)
    rng_b = np.random.default_rng(seed + 2)
    s_a = Substrate.fresh(seed=seed + 100)
    s_b = Substrate.fresh(seed=seed + 200)
    traj_a = wake(s_a, world, n_wake, rng_a)
    traj_b = wake(s_b, world, n_wake, rng_b)
    sleep(s_a, n_replay, rng_a, wipe=True)
    sleep(s_b, n_replay, rng_b, wipe=True)

    def predict_next(s, flavor):
        return int(np.argmax(s.W_trans.sum(axis=1)[flavor]))

    def acc(s, traj):
        if len(traj) < 2:
            return 0.0
        transitions = {}
        for i in range(len(traj) - 1):
            transitions.setdefault(traj[i][1], []).append(traj[i + 1][1])
        if not transitions:
            return 0.0
        correct = 0
        for f_t, next_list in transitions.items():
            modal = max(set(next_list), key=next_list.count)
            if predict_next(s, f_t) == modal:
                correct += 1
        return correct / len(transitions)

    return {
        "a_own": acc(s_a, traj_a), "a_other": acc(s_a, traj_b),
        "b_own": acc(s_b, traj_b), "b_other": acc(s_b, traj_a),
    }


# --- Main ----------------------------------------------------------------

def main():
    print("=" * 72)
    print(f"IDENTITY TESTS v3 — bigger world ({N_POSITIONS}p × {N_FLAVORS}f), no personality")
    print("=" * 72)

    print("\n--- TEST 1: Behavioral continuity ---")
    sims = [behavioral_continuity(seed) for seed in range(10)]
    for i, s in enumerate(sims):
        print(f"  Seed {i}: {s:.4f}")
    print(f"  Mean: {np.mean(sims):.4f}   Min: {np.min(sims):.4f}   Max: {np.max(sims):.4f}")

    print("\n--- TEST 2A: Same world ---")
    r2a = self_vs_other(same_world=True)
    print(f"  Strict self-ID: {r2a['correct']}/{r2a['n']}")
    print(f"  Self-sim {r2a['mean_self']:.4f}, other-sim {r2a['mean_other']:.4f}, gap +{r2a['gap']:.4f}")

    print("\n--- TEST 2B: Different worlds ---")
    r2b = self_vs_other(same_world=False)
    print(f"  Strict self-ID: {r2b['correct']}/{r2b['n']}")
    print(f"  Self-sim {r2b['mean_self']:.4f}, other-sim {r2b['mean_other']:.4f}, gap +{r2b['gap']:.4f}")

    print("\n--- TEST 3: Episode-specific recall ---")
    test3 = [episode_specific_recall(seed) for seed in range(10)]
    n_pass_a = sum(1 for r in test3 if r['a_own'] > r['a_other'] + 0.05)
    n_pass_b = sum(1 for r in test3 if r['b_own'] > r['b_other'] + 0.05)
    mean_a_gap = np.mean([r['a_own'] - r['a_other'] for r in test3])
    mean_b_gap = np.mean([r['b_own'] - r['b_other'] for r in test3])
    for i, r in enumerate(test3):
        a_p = "PASS" if r['a_own'] > r['a_other'] + 0.05 else "fail"
        b_p = "PASS" if r['b_own'] > r['b_other'] + 0.05 else "fail"
        print(f"  Seed {i}: A {r['a_own']:.2f}/{r['a_other']:.2f} ({a_p})   B {r['b_own']:.2f}/{r['b_other']:.2f} ({b_p})")
    print(f"\n  A: {n_pass_a}/10 mean gap +{mean_a_gap:.3f}")
    print(f"  B: {n_pass_b}/10 mean gap +{mean_b_gap:.3f}")
    print(f"  Total: {n_pass_a + n_pass_b}/20")

    print(f"\n{'=' * 72}")
    print("v3 SUMMARY (bigger world, no personality)")
    print(f"{'=' * 72}")
    print(f"  Test 1 — continuity:           {np.mean(sims):.4f}")
    print(f"  Test 2A — same world:          {r2a['correct']}/10  gap +{r2a['gap']:.3f}")
    print(f"  Test 2B — different worlds:    {r2b['correct']}/10  gap +{r2b['gap']:.3f}")
    print(f"  Test 3 — episode-specific:     {n_pass_a + n_pass_b}/20")


if __name__ == "__main__":
    main()
