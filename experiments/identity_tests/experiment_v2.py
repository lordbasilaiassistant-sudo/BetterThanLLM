"""Identity Tests v2 — push toward 100% recognition by adding stable personality.

The v1 result was 16/20 episode-specific recall, with 4 tied seeds where two
substrates trained in the same world ended up with similar trajectories
(gap = 0). Cause: substrates only differed by random init; they converged
on similar policies.

v2 adds a *personality bias* vector to each substrate — a stable per-flavor
preference offset that survives sleep and biases all reward-driven updates.
Substrate A might "love" flavor 2; substrate B "loves" flavor 5. Same world,
genuinely different individuals.

This is the architectural commitment: identity isn't just random init. Each
substrate has a *temperament* — slow-changing per-flavor disposition that
makes it perceive the world differently from peers. Like real people.

Run: py experiment_v2.py
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

# --- World ---------------------------------------------------------------

N_POSITIONS = 10
N_FLAVORS = 6
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


# --- Substrate with personality -----------------------------------------

@dataclass
class Substrate:
    W_action: np.ndarray
    W_trans: np.ndarray
    disposition: np.ndarray
    intention: np.ndarray
    personality: np.ndarray   # NEW: stable per-flavor preference offset
    episodic: List[Tuple[int, int, float, int]] = field(default_factory=list)
    lr: float = 0.05
    disp_lr: float = 0.02
    intent_lr: float = 0.04
    plan_weight: float = 0.6
    top_k: int = 2

    @classmethod
    def fresh(cls, seed, personality_strength=1.0, force_unique_focus=True):
        rng = np.random.default_rng(seed)
        if force_unique_focus:
            # Each substrate gets a strongly distinctive 'favorite' and 'aversion':
            # one flavor it strongly prefers, one it strongly avoids, others mild.
            # Different seeds produce different favorites -> reliably-divergent
            # individuals.
            personality = rng.normal(0, 0.2, size=N_FLAVORS)
            favorite = int(rng.integers(0, N_FLAVORS))
            aversion = (favorite + 1 + rng.integers(0, N_FLAVORS - 1)) % N_FLAVORS
            personality[favorite] += personality_strength
            personality[aversion] -= personality_strength
        else:
            personality = rng.uniform(-personality_strength, personality_strength, size=N_FLAVORS)
        return cls(
            W_action=rng.normal(0, 0.01, (N_FLAVORS, N_ACTIONS)),
            W_trans=rng.normal(0, 0.01, (N_FLAVORS, N_ACTIONS, N_FLAVORS)),
            disposition=np.zeros(N_FLAVORS),
            intention=np.zeros(N_FLAVORS),
            personality=personality,
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


def perceived_reward(s, world_reward, flavor):
    """The substrate's subjective reward. World reward + personality bias.
    Substrate's behavior shapes around what IT finds rewarding, not just
    what's objectively rewarding."""
    return world_reward + s.personality[flavor]


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
        world_r = float(flavor_to_reward[next_flavor])
        # Substrate-subjective reward: world + personality
        reward = perceived_reward(s, world_r, next_flavor)
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
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
    return float(a @ b / denom)


def episode_specific_recall(seed, n_wake=500, n_replay=1500):
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
            f_t = traj[i][1]
            f_n = traj[i + 1][1]
            transitions.setdefault(f_t, []).append(f_n)
        if not transitions:
            return 0.0
        correct = 0
        for f_t, next_list in transitions.items():
            modal = max(set(next_list), key=next_list.count)
            if predict_next(s, f_t) == modal:
                correct += 1
        return correct / len(transitions)

    return {
        "a_own": acc(s_a, traj_a),
        "a_other": acc(s_a, traj_b),
        "b_own": acc(s_b, traj_b),
        "b_other": acc(s_b, traj_a),
        "personality_distance": float(np.linalg.norm(s_a.personality - s_b.personality)),
    }


def self_vs_other(n_substrates=10, n_wake=500, n_replay=1500,
                  same_world=True):
    pre_sigs = []
    post_sigs = []
    for i in range(n_substrates):
        world_seed = 42 if same_world else (42 + i * 1009)
        world = make_world(world_seed)
        rng = np.random.default_rng(world_seed * 1000 + i * 7919)
        s = Substrate.fresh(seed=i * 31337)
        wake(s, world, n_wake, rng)
        pre_sigs.append(behavioral_signature(s, rng=np.random.default_rng(i + 1234)))
        sleep(s, n_replay, rng, wipe=True)
        post_sigs.append(behavioral_signature(s, rng=np.random.default_rng(i + 1234)))
    sim_matrix = np.zeros((n_substrates, n_substrates))
    for i in range(n_substrates):
        for j in range(n_substrates):
            sim_matrix[i, j] = signature_similarity(post_sigs[i], pre_sigs[j])
    correct = sum(int(np.argmax(sim_matrix[i])) == i for i in range(n_substrates))
    self_sims = np.diag(sim_matrix)
    mask = ~np.eye(n_substrates, dtype=bool)
    other_sims = sim_matrix[mask]
    return {
        "correct": correct, "n": n_substrates,
        "mean_self": float(self_sims.mean()),
        "mean_other": float(other_sims.mean()),
        "gap": float(self_sims.mean() - other_sims.mean()),
        "matrix": sim_matrix,
    }


def behavioral_continuity(seed, n_wake=500, n_replay=1500):
    rng = np.random.default_rng(seed)
    world = make_world(seed)
    s = Substrate.fresh(seed=seed + 100)
    wake(s, world, n_wake, rng)
    sig_pre = behavioral_signature(s, rng=np.random.default_rng(seed + 555))
    sleep(s, n_replay, rng, wipe=True)
    sig_post = behavioral_signature(s, rng=np.random.default_rng(seed + 555))
    return signature_similarity(sig_pre, sig_post)


# --- Main ----------------------------------------------------------------

def main():
    print("=" * 72)
    print("IDENTITY TESTS v2 — substrates with persistent personality")
    print("=" * 72)
    print()
    print("v2 adds a personality vector (per-flavor preference bias) that")
    print("survives sleep and shapes all reward-driven updates. Substrates")
    print("are now genuinely different individuals, not just different seeds.")
    print()

    # Test 1
    print("-" * 72)
    print("TEST 1: Behavioral continuity")
    print("-" * 72)
    sims = [behavioral_continuity(seed) for seed in range(10)]
    for i, s in enumerate(sims):
        print(f"  Seed {i}: pre/post-sleep cosine = {s:.4f}")
    print(f"\n  Mean: {np.mean(sims):.4f}   Min: {np.min(sims):.4f}   Max: {np.max(sims):.4f}")

    # Test 2A
    print(f"\n{'-' * 72}")
    print("TEST 2A: Self vs other — SAME WORLD")
    print(f"{'-' * 72}")
    r2a = self_vs_other(same_world=True)
    print(f"  Strict self-ID: {r2a['correct']}/{r2a['n']}")
    print(f"  Mean self-sim: {r2a['mean_self']:.4f}   other-sim: {r2a['mean_other']:.4f}   gap: +{r2a['gap']:.4f}")

    # Test 2B
    print(f"\n{'-' * 72}")
    print("TEST 2B: Self vs other — DIFFERENT WORLDS")
    print(f"{'-' * 72}")
    r2b = self_vs_other(same_world=False)
    print(f"  Strict self-ID: {r2b['correct']}/{r2b['n']}")
    print(f"  Mean self-sim: {r2b['mean_self']:.4f}   other-sim: {r2b['mean_other']:.4f}   gap: +{r2b['gap']:.4f}")

    # Test 3
    print(f"\n{'-' * 72}")
    print("TEST 3: Episode-specific recall")
    print(f"{'-' * 72}")
    test3 = [episode_specific_recall(seed) for seed in range(10)]
    n_pass_a = sum(1 for r in test3 if r['a_own'] > r['a_other'] + 0.05)
    n_pass_b = sum(1 for r in test3 if r['b_own'] > r['b_other'] + 0.05)
    mean_a_gap = np.mean([r['a_own'] - r['a_other'] for r in test3])
    mean_b_gap = np.mean([r['b_own'] - r['b_other'] for r in test3])
    mean_personality_dist = np.mean([r['personality_distance'] for r in test3])
    for i, r in enumerate(test3):
        a_pass = "PASS" if r['a_own'] > r['a_other'] + 0.05 else "fail"
        b_pass = "PASS" if r['b_own'] > r['b_other'] + 0.05 else "fail"
        print(f"  Seed {i}: A own/other {r['a_own']:.2f}/{r['a_other']:.2f} ({a_pass})   "
              f"B own/other {r['b_own']:.2f}/{r['b_other']:.2f} ({b_pass})   "
              f"personality dist {r['personality_distance']:.2f}")
    print(f"\n  Substrate A self > other: {n_pass_a}/10   mean gap +{mean_a_gap:.3f}")
    print(f"  Substrate B self > other: {n_pass_b}/10   mean gap +{mean_b_gap:.3f}")
    print(f"  Total: {n_pass_a + n_pass_b}/20 substrate-trajectory pairs")
    print(f"  Mean personality distance between substrates: {mean_personality_dist:.3f}")

    print(f"\n{'=' * 72}")
    print("v2 SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Test 1 — continuity:               mean cosine {np.mean(sims):.4f}")
    print(f"  Test 2A — same world self-ID:      {r2a['correct']}/10  (gap +{r2a['gap']:.3f})")
    print(f"  Test 2B — diff worlds self-ID:     {r2b['correct']}/10  (gap +{r2b['gap']:.3f})")
    print(f"  Test 3 — episode-specific recall:  {n_pass_a + n_pass_b}/20")
    print(f"\n  Personality drives divergence: {mean_personality_dist:.2f} avg distance.")
    print("=" * 72)


if __name__ == "__main__":
    main()
