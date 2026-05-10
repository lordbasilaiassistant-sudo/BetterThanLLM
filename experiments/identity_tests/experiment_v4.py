"""Identity Tests v4 — push every avenue.

Six tests, batched. Each tests a different facet of substrate-identity that
LLMs cannot have. World is now 30 positions × 12 flavors — the biggest yet,
giving substrates room to genuinely diverge.

  T1. Behavioral continuity            (single sleep)
  T2. Multi-cycle identity              (10 sleeps; does identity stay coherent across many gaps?)
  T3. Self vs other (different worlds) (clean individuation)
  T4. Episode-specific recall          (push toward 20/20)
  T5. Counterfactual fork              (novel: fork substrate at time T, both forks remember pre-fork past?)
  T6. Identity transfer                (copy weights to fresh container — behavior identical?)
  T7. Adversarial damage               (zero 30% of W_trans; does substrate retain core identity?)

Run: py experiment_v4.py
"""

import numpy as np
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple

# --- World ---------------------------------------------------------------

N_POSITIONS = 30
N_FLAVORS = 12
N_ACTIONS = 3


def make_world(seed):
    rng = np.random.default_rng(seed)
    pos_to_flavor = rng.integers(0, N_FLAVORS, size=N_POSITIONS)
    flavor_to_reward = rng.uniform(-1, 1, size=N_FLAVORS)
    if flavor_to_reward.max() < 0.5:
        flavor_to_reward[int(np.argmax(flavor_to_reward))] = float(rng.uniform(0.5, 1.0))
    return pos_to_flavor, flavor_to_reward


def step_world(pos, action):
    if action == 0: return max(0, pos - 1)
    if action == 1: return min(N_POSITIONS - 1, pos + 1)
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

    def deep_clone(self):
        c = Substrate.fresh(seed=0)
        c.W_action = self.W_action.copy()
        c.W_trans = self.W_trans.copy()
        c.disposition = self.disposition.copy()
        c.intention = self.intention.copy()
        return c


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


# --- Identity helpers ---------------------------------------------------

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
    a, b = sig_a.flatten(), sig_b.flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# --- T1: Behavioral continuity ------------------------------------------

def t1_behavioral_continuity(seed, n_wake=800, n_replay=1500):
    rng = np.random.default_rng(seed)
    world = make_world(seed)
    s = Substrate.fresh(seed=seed + 100)
    wake(s, world, n_wake, rng)
    sig_pre = behavioral_signature(s, rng=np.random.default_rng(seed + 555))
    sleep(s, n_replay, rng, wipe=True)
    sig_post = behavioral_signature(s, rng=np.random.default_rng(seed + 555))
    return signature_similarity(sig_pre, sig_post)


# --- T2: Multi-cycle identity -------------------------------------------

def t2_multi_cycle(seed, n_cycles=10, n_wake=300, n_replay=800):
    """10 wake/sleep cycles in same world. Track signature similarity to
    cycle 1 across all subsequent cycles. Pass: substrate doesn't drift
    away from original identity over many cycles.
    """
    rng = np.random.default_rng(seed)
    world = make_world(seed)
    s = Substrate.fresh(seed=seed + 100)

    sigs = []
    for cycle in range(n_cycles):
        wake(s, world, n_wake, rng)
        sleep(s, n_replay, rng, wipe=True)
        sigs.append(behavioral_signature(s, rng=np.random.default_rng(seed + cycle + 555)))

    # Similarity of cycle i to cycle 0 (original substrate)
    sims_to_first = [signature_similarity(sigs[0], sigs[i]) for i in range(n_cycles)]
    # Day-to-day similarity
    d2d = [signature_similarity(sigs[i], sigs[i + 1]) for i in range(n_cycles - 1)]
    return {
        "sims_to_first": sims_to_first,
        "d2d_sims": d2d,
        "final_to_first": sims_to_first[-1],
        "min_d2d": min(d2d),
    }


# --- T3: Self vs other (different worlds) -------------------------------

def t3_self_vs_other(n_substrates=10, n_wake=800, n_replay=1500, same_world=False):
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
    sim = np.array([[signature_similarity(post_sigs[i], pre_sigs[j])
                     for j in range(n_substrates)] for i in range(n_substrates)])
    correct = sum(int(np.argmax(sim[i])) == i for i in range(n_substrates))
    self_sims = np.diag(sim)
    other_sims = sim[~np.eye(n_substrates, dtype=bool)]
    return {
        "correct": correct, "n": n_substrates,
        "mean_self": float(self_sims.mean()),
        "mean_other": float(other_sims.mean()),
    }


# --- T4: Episode-specific recall ----------------------------------------

def t4_episode_recall(seed, n_wake=800, n_replay=1500):
    world = make_world(seed)
    rng_a = np.random.default_rng(seed + 1)
    rng_b = np.random.default_rng(seed + 2)
    s_a = Substrate.fresh(seed=seed + 100)
    s_b = Substrate.fresh(seed=seed + 200)
    traj_a = wake(s_a, world, n_wake, rng_a)
    traj_b = wake(s_b, world, n_wake, rng_b)
    sleep(s_a, n_replay, rng_a, wipe=True)
    sleep(s_b, n_replay, rng_b, wipe=True)

    def predict(s, f): return int(np.argmax(s.W_trans.sum(axis=1)[f]))

    def acc(s, traj):
        if len(traj) < 2: return 0.0
        transitions = {}
        for i in range(len(traj) - 1):
            transitions.setdefault(traj[i][1], []).append(traj[i + 1][1])
        if not transitions: return 0.0
        return sum(predict(s, f) == max(set(nl), key=nl.count)
                   for f, nl in transitions.items()) / len(transitions)

    return {"a_own": acc(s_a, traj_a), "a_other": acc(s_a, traj_b),
            "b_own": acc(s_b, traj_b), "b_other": acc(s_b, traj_a)}


# --- T5: Counterfactual fork --------------------------------------------

def t5_counterfactual_fork(seed, n_wake_pre=400, n_replay_pre=800,
                           n_wake_fork=200, n_replay_fork=400):
    """Train substrate on shared prefix. Fork into A and B, each runs in
    different worlds. After sleep, each fork should:
      (a) Remember the shared pre-fork past better than a fresh substrate
      (b) Be distinguishable from each other (post-fork divergence is real)
    """
    rng = np.random.default_rng(seed)
    world_pre = make_world(seed)
    world_a = make_world(seed + 5000)
    world_b = make_world(seed + 9000)

    # Pre-fork: shared substrate runs in world_pre
    s_pre = Substrate.fresh(seed=seed + 100)
    traj_pre = wake(s_pre, world_pre, n_wake_pre, rng)
    sleep(s_pre, n_replay_pre, rng, wipe=True)

    # Snapshot pre-fork signature for "remembers shared past" test
    sig_pre = behavioral_signature(s_pre, rng=np.random.default_rng(seed + 333))

    # Fork
    s_a = s_pre.deep_clone()
    s_b = s_pre.deep_clone()

    # Each fork runs in its own world
    rng_a = np.random.default_rng(seed + 7777)
    rng_b = np.random.default_rng(seed + 8888)
    wake(s_a, world_a, n_wake_fork, rng_a)
    sleep(s_a, n_replay_fork, rng_a, wipe=True)
    wake(s_b, world_b, n_wake_fork, rng_b)
    sleep(s_b, n_replay_fork, rng_b, wipe=True)

    # Post-fork signatures
    sig_a = behavioral_signature(s_a, rng=np.random.default_rng(seed + 333))
    sig_b = behavioral_signature(s_b, rng=np.random.default_rng(seed + 333))

    # (a) Each fork should remember pre-fork: sig_a vs sig_pre, sig_b vs sig_pre
    a_to_pre = signature_similarity(sig_a, sig_pre)
    b_to_pre = signature_similarity(sig_b, sig_pre)

    # (b) Forks should be distinguishable from each other
    a_to_b = signature_similarity(sig_a, sig_b)

    # Control: fresh substrate (never had pre-fork experience) signature
    s_fresh = Substrate.fresh(seed=seed + 999)
    sig_fresh = behavioral_signature(s_fresh, rng=np.random.default_rng(seed + 333))
    fresh_to_pre = signature_similarity(sig_fresh, sig_pre)

    return {
        "a_to_pre": a_to_pre,
        "b_to_pre": b_to_pre,
        "a_to_b": a_to_b,
        "fresh_to_pre": fresh_to_pre,
        "shared_past_signal": float((a_to_pre + b_to_pre) / 2 - fresh_to_pre),
        "fork_divergence": float((a_to_pre + b_to_pre) / 2 - a_to_b),
    }


# --- T6: Identity transfer (copy weights) -------------------------------

def t6_identity_transfer(seed, n_wake=500, n_replay=1000):
    """Train substrate. Copy weights to a fresh container. Test if both
    behave identically. The trivial-but-needs-verifying claim: identity
    is in the weights, so an exact copy IS the same individual.
    """
    rng = np.random.default_rng(seed)
    world = make_world(seed)
    s_orig = Substrate.fresh(seed=seed + 100)
    wake(s_orig, world, n_wake, rng)
    sleep(s_orig, n_replay, rng, wipe=True)

    s_copy = s_orig.deep_clone()

    sig_orig = behavioral_signature(s_orig, rng=np.random.default_rng(seed + 333))
    sig_copy = behavioral_signature(s_copy, rng=np.random.default_rng(seed + 333))

    return signature_similarity(sig_orig, sig_copy)


# --- T7: Adversarial damage ---------------------------------------------

def t7_adversarial_damage(seed, n_wake=500, n_replay=1000, damage_frac=0.3):
    """Train substrate. Zero out `damage_frac` fraction of W_trans entries
    (random). Compare signature before vs after. Substrate should retain
    SOME identity — graceful degradation, not catastrophic loss.
    """
    rng = np.random.default_rng(seed)
    world = make_world(seed)
    s = Substrate.fresh(seed=seed + 100)
    wake(s, world, n_wake, rng)
    sleep(s, n_replay, rng, wipe=True)

    sig_pre = behavioral_signature(s, rng=np.random.default_rng(seed + 333))

    # Damage
    n_total = s.W_trans.size
    n_damage = int(damage_frac * n_total)
    flat = s.W_trans.flatten()
    damage_idx = rng.choice(n_total, n_damage, replace=False)
    flat[damage_idx] = 0.0
    s.W_trans = flat.reshape(s.W_trans.shape)

    sig_post = behavioral_signature(s, rng=np.random.default_rng(seed + 333))

    return signature_similarity(sig_pre, sig_post)


# --- Main ----------------------------------------------------------------

def main():
    print("=" * 72)
    print(f"IDENTITY TESTS v4 — every avenue, world {N_POSITIONS}p × {N_FLAVORS}f")
    print("=" * 72)

    # T1
    print("\n--- T1: Single-sleep behavioral continuity ---")
    t1 = [t1_behavioral_continuity(s) for s in range(10)]
    print(f"  Mean cosine: {np.mean(t1):.4f}   (range {min(t1):.3f} - {max(t1):.3f})")

    # T2
    print("\n--- T2: Multi-cycle identity (10 sleeps) ---")
    t2_results = [t2_multi_cycle(s, n_cycles=10) for s in range(5)]
    final_sims = [r['final_to_first'] for r in t2_results]
    min_d2d = [r['min_d2d'] for r in t2_results]
    print(f"  Mean cycle-10 vs cycle-1 sim: {np.mean(final_sims):.4f}")
    print(f"  Mean min day-to-day similarity: {np.mean(min_d2d):.4f}")
    # Show first seed's trajectory
    r0 = t2_results[0]
    print(f"  Seed 0 sims to cycle 1: {[f'{s:.3f}' for s in r0['sims_to_first']]}")

    # T3
    print("\n--- T3: Self vs other (different worlds) ---")
    t3 = t3_self_vs_other(same_world=False)
    print(f"  Strict self-ID: {t3['correct']}/{t3['n']}")
    print(f"  Mean self-sim {t3['mean_self']:.3f}, other-sim {t3['mean_other']:.3f}, gap +{t3['mean_self'] - t3['mean_other']:.3f}")

    # T4
    print("\n--- T4: Episode-specific recall ---")
    t4_results = [t4_episode_recall(s) for s in range(10)]
    n_pass_a = sum(1 for r in t4_results if r['a_own'] > r['a_other'] + 0.05)
    n_pass_b = sum(1 for r in t4_results if r['b_own'] > r['b_other'] + 0.05)
    mean_a_gap = np.mean([r['a_own'] - r['a_other'] for r in t4_results])
    mean_b_gap = np.mean([r['b_own'] - r['b_other'] for r in t4_results])
    for i, r in enumerate(t4_results):
        ap = "P" if r['a_own'] > r['a_other'] + 0.05 else "."
        bp = "P" if r['b_own'] > r['b_other'] + 0.05 else "."
        print(f"  Seed {i}: A {r['a_own']:.2f}/{r['a_other']:.2f}({ap})  B {r['b_own']:.2f}/{r['b_other']:.2f}({bp})")
    print(f"  Total: {n_pass_a + n_pass_b}/20   mean gap A={mean_a_gap:+.3f}  B={mean_b_gap:+.3f}")

    # T5
    print("\n--- T5: Counterfactual fork ---")
    t5 = [t5_counterfactual_fork(s) for s in range(10)]
    mean_a_to_pre = np.mean([r['a_to_pre'] for r in t5])
    mean_b_to_pre = np.mean([r['b_to_pre'] for r in t5])
    mean_a_to_b = np.mean([r['a_to_b'] for r in t5])
    mean_fresh_to_pre = np.mean([r['fresh_to_pre'] for r in t5])
    mean_shared_signal = np.mean([r['shared_past_signal'] for r in t5])
    mean_fork_div = np.mean([r['fork_divergence'] for r in t5])
    print(f"  Mean fork-A vs pre-fork: {mean_a_to_pre:.4f}")
    print(f"  Mean fork-B vs pre-fork: {mean_b_to_pre:.4f}")
    print(f"  Mean fork-A vs fork-B:   {mean_a_to_b:.4f}")
    print(f"  Mean fresh vs pre-fork:  {mean_fresh_to_pre:.4f}")
    print(f"  Shared-past signal (forks vs pre, vs fresh vs pre): {mean_shared_signal:+.4f}")
    print(f"  Fork divergence (forks remember shared past more than each other): {mean_fork_div:+.4f}")

    # T6
    print("\n--- T6: Identity transfer (deep clone) ---")
    t6 = [t6_identity_transfer(s) for s in range(10)]
    print(f"  Mean orig-vs-copy similarity: {np.mean(t6):.4f}   (should be ~1.0)")

    # T7
    print("\n--- T7: Adversarial damage (30% W_trans zeroed) ---")
    t7 = [t7_adversarial_damage(s) for s in range(10)]
    print(f"  Mean pre-damage vs post-damage similarity: {np.mean(t7):.4f}")
    print(f"  Range: {min(t7):.3f} - {max(t7):.3f}")

    print(f"\n{'=' * 72}")
    print("v4 SUMMARY")
    print(f"{'=' * 72}")
    print(f"  T1 (single-sleep continuity):      {np.mean(t1):.4f}")
    print(f"  T2 (10-cycle, cycle-10 vs cycle-1):{np.mean(final_sims):.4f}")
    print(f"  T3 (self vs other diff worlds):    {t3['correct']}/10")
    print(f"  T4 (episode-specific):             {n_pass_a + n_pass_b}/20")
    print(f"  T5 (fork: shared-past signal):     {mean_shared_signal:+.3f}  (positive = forks remember)")
    print(f"  T5 (fork: divergence signal):      {mean_fork_div:+.3f}  (positive = forks distinct)")
    print(f"  T6 (identity transfer):            {np.mean(t6):.4f}  (should be 1.0)")
    print(f"  T7 (adversarial damage retention): {np.mean(t7):.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
