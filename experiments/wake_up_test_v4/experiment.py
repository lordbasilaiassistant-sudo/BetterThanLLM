"""Wake-Up Test v4 — characterize architecture trade-offs honestly.

The v3 fact-check showed substrate underperforms trivial frequency counting
in toy regimes. v4 asks two specific questions:

  1. CAN the substrate match counting if we drop the "biological" features
     (Hebbian decay, noisy replay)? If yes, those features are the cost
     of biological plausibility, and the architecture has parity with
     counting at parity settings.

  2. IS THERE ANY REGIME where substrate beats counting?
     Test: NON-STATIONARY world. Rewards drift over time. Counting
     accumulates stale data; substrate's decay forgets it. If substrate
     adapts faster to current rewards, that IS a win — decay is a
     feature, not a bug, for non-stationary worlds.

This is honest characterization. We're not trying to manufacture a win;
we're mapping where the architecture lives in the cost/capability space.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wake_up_test_v3.experiment import (
    Substrate, make_world, step_world, select_action,
    softmax, softmax_topk,
    N_POSITIONS, N_FLAVORS, N_ACTIONS,
)
from wake_up_test_v3.fact_check import (
    trivial_count_table, predict_from_counts, autobiographical_acc,
)


# --- Tunable hebb_update ------------------------------------------------

def hebb_update_tunable(s, flavor, action, reward, next_flavor,
                       use_decay=True, allow_intent=True):
    """Hebbian update with per-feature toggles for ablation."""
    s.W_action[flavor, action] += s.lr * reward
    s.W_trans[flavor, action, next_flavor] += s.lr
    if use_decay:
        s.W_action[flavor] *= (1 - s.lr * 0.05)
        s.W_trans[flavor, action] *= (1 - s.lr * 0.05)
    s.disposition[flavor] += s.disp_lr * (reward - s.disposition[flavor])
    if allow_intent:
        baseline = float(s.disposition.mean())
        relative = reward - baseline
        if relative > 0.05:
            s.intention[next_flavor] += s.intent_lr * relative
            if use_decay:
                s.intention *= (1 - s.intent_lr * 0.05)


def wake_tunable(s, world, n_steps, rng, use_decay=True):
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
        hebb_update_tunable(s, flavor, action, reward, next_flavor, use_decay=use_decay)
        pos = next_pos
    return trajectory


def sleep_tunable(s, n_replay, rng, noise=0.05, use_decay=True, wipe=True):
    if not s.episodic:
        return
    n = len(s.episodic)
    for _ in range(n_replay):
        flavor, action, reward, next_flavor = s.episodic[rng.integers(0, n)]
        if noise > 0 and rng.random() < noise:
            flavor = int((flavor + 1 + rng.integers(0, N_FLAVORS - 1)) % N_FLAVORS)
        hebb_update_tunable(s, flavor, action, reward, next_flavor, use_decay=use_decay)
    if wipe:
        s.episodic = []


# --- Question 1: parameter sweep on autobiographical task --------------

def run_config(seed, use_decay, replay_noise, n_wake=300, n_replay=800):
    rng = np.random.default_rng(seed)
    world = make_world(seed)
    s = Substrate.fresh(seed=seed + 100)
    traj = wake_tunable(s, world, n_wake, rng, use_decay=use_decay)
    sleep_tunable(s, n_replay, rng, noise=replay_noise, use_decay=use_decay, wipe=True)
    marginal = s.W_trans.sum(axis=1)
    pred = lambda f: int(np.argmax(marginal[f]))
    return autobiographical_acc(pred, traj), traj


def question_1_param_sweep():
    print("=" * 72)
    print("QUESTION 1: Can substrate match counting with biological features off?")
    print("=" * 72)
    print()

    configs = [
        ("decay=ON,  noise=0.05  (v3 default)", True, 0.05),
        ("decay=ON,  noise=0     (no replay noise)", True, 0.0),
        ("decay=OFF, noise=0.05  (no decay)", False, 0.05),
        ("decay=OFF, noise=0     (parity baseline)", False, 0.0),
    ]

    n_seeds = 10
    print(f"{'Config':<45}  {'Sub acc':<10}  {'Counts acc':<10}  {'Gap':<8}")
    print("-" * 80)

    for label, decay, noise in configs:
        sub_accs = []
        counts_accs = []
        for seed in range(n_seeds):
            sub_acc, traj = run_config(seed, decay, noise)
            counts = trivial_count_table(traj)
            counts_acc = autobiographical_acc(lambda f: predict_from_counts(counts, f), traj)
            sub_accs.append(sub_acc)
            counts_accs.append(counts_acc)
        mean_sub = np.mean(sub_accs)
        mean_counts = np.mean(counts_accs)
        gap = mean_sub - mean_counts
        print(f"{label:<45}  {mean_sub:<10.3f}  {mean_counts:<10.3f}  {gap:+.3f}")

    print()
    print("Interpretation:")
    print("  - decay+noise: full v3 substrate, expected to underperform counts.")
    print("  - decay only:  decay alone causes slow forgetting; loses info.")
    print("  - noise only:  noise alone scrambles ~5% of replays; small effect.")
    print("  - parity:      no biological features; should match counts.")


# --- Question 2: non-stationary world ------------------------------------

def make_drifting_world(seed, drift_period=100, n_episodes=600):
    """A world whose flavor-position assignment shifts every drift_period
    steps. Tracks WHICH world is 'current' at each step.

    Returns:
      worlds: list of (pos_to_flavor, flavor_to_reward) tuples
      schedule: list of length n_episodes giving which world index is
                 active at each step
    """
    rng = np.random.default_rng(seed)
    n_worlds = (n_episodes + drift_period - 1) // drift_period
    worlds = []
    for w in range(n_worlds):
        pos_to_flavor = rng.integers(0, N_FLAVORS, size=N_POSITIONS)
        flavor_to_reward = rng.uniform(-1, 1, size=N_FLAVORS)
        if flavor_to_reward.max() < 0.5:
            flavor_to_reward[int(np.argmax(flavor_to_reward))] = float(rng.uniform(0.5, 1.0))
        worlds.append((pos_to_flavor, flavor_to_reward))
    schedule = [min(t // drift_period, n_worlds - 1) for t in range(n_episodes)]
    return worlds, schedule


def wake_drifting(s, worlds, schedule, rng, use_decay=True):
    """Wake on a drifting world. Each step uses worlds[schedule[t]]."""
    pos = int(rng.integers(0, N_POSITIONS))
    trajectory = []
    for t, w_idx in enumerate(schedule):
        pos_to_flavor, flavor_to_reward = worlds[w_idx]
        flavor = int(pos_to_flavor[pos])
        action, _ = select_action(s, flavor, rng)
        next_pos = step_world(pos, action)
        next_flavor = int(pos_to_flavor[next_pos])
        reward = float(flavor_to_reward[next_flavor])
        s.episodic.append((flavor, action, reward, next_flavor))
        trajectory.append((pos, flavor, action, reward, next_pos, w_idx))
        hebb_update_tunable(s, flavor, action, reward, next_flavor, use_decay=use_decay)
        pos = next_pos
    return trajectory


def question_2_nonstationary():
    print()
    print("=" * 72)
    print("QUESTION 2: Does substrate's decay help in non-stationary worlds?")
    print("=" * 72)
    print()
    print("Setup: world drifts every 100 steps (transitions reassigned).")
    print("After 600 steps + sleep + wipe, predict CURRENT-world transitions.")
    print()
    print("Substrate (with decay): old pairs decay, recent pairs dominate -> tracks current.")
    print("Counts (no decay):      all pairs accumulated -> dominated by stale data.")
    print()

    n_seeds = 10
    drift_period = 100
    n_episodes = 600

    print(f"{'Config':<35}  {'Sub acc':<10}  {'Counts acc':<10}  {'Gap':<8}")
    print("-" * 70)

    sub_accs_decay = []
    sub_accs_no_decay = []
    counts_accs = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        worlds, schedule = make_drifting_world(seed, drift_period=drift_period, n_episodes=n_episodes)
        current_world = worlds[schedule[-1]]

        # Substrate WITH decay
        s_decay = Substrate.fresh(seed=seed + 100)
        traj = wake_drifting(s_decay, worlds, schedule, rng, use_decay=True)
        sleep_tunable(s_decay, 800, rng, noise=0.05, use_decay=True, wipe=True)

        # Substrate WITHOUT decay (parity baseline)
        rng2 = np.random.default_rng(seed)
        s_no_decay = Substrate.fresh(seed=seed + 100)
        traj2 = wake_drifting(s_no_decay, worlds, schedule, rng2, use_decay=False)
        sleep_tunable(s_no_decay, 800, rng2, noise=0.0, use_decay=False, wipe=True)

        # Test on CURRENT world's true transitions
        # Build a target transition table from current world by simulating clean transitions
        true_transitions = {}
        pos_to_flavor, _ = current_world
        for p in range(N_POSITIONS):
            f = int(pos_to_flavor[p])
            for a in range(N_ACTIONS):
                np_pos = step_world(p, a)
                nf = int(pos_to_flavor[np_pos])
                true_transitions.setdefault(f, []).append(nf)
        # Modal next-flavor per flavor in current world
        target = {f: max(set(nl), key=nl.count) for f, nl in true_transitions.items()}

        marg_d = s_decay.W_trans.sum(axis=1)
        marg_nd = s_no_decay.W_trans.sum(axis=1)
        sub_d_acc = sum(int(np.argmax(marg_d[f])) == m for f, m in target.items()) / len(target)
        sub_nd_acc = sum(int(np.argmax(marg_nd[f])) == m for f, m in target.items()) / len(target)
        sub_accs_decay.append(sub_d_acc)
        sub_accs_no_decay.append(sub_nd_acc)

        # Counts using all pairs in trajectory (no decay possible)
        counts = trivial_count_table([(p, f, a, r, nf) for (p, f, a, r, nf, _) in traj])
        counts_acc = sum(int(np.argmax(counts[f])) == m for f, m in target.items() if counts[f].sum() > 0) / len(target)
        counts_accs.append(counts_acc)

    mean_sd = np.mean(sub_accs_decay)
    mean_snd = np.mean(sub_accs_no_decay)
    mean_c = np.mean(counts_accs)

    print(f"{'Substrate (decay ON)':<35}  {mean_sd:<10.3f}  {mean_c:<10.3f}  {mean_sd - mean_c:+.3f}")
    print(f"{'Substrate (decay OFF, parity)':<35}  {mean_snd:<10.3f}  {mean_c:<10.3f}  {mean_snd - mean_c:+.3f}")
    print()
    print(f"Decay vs no-decay substrate gap: {mean_sd - mean_snd:+.3f}")
    print()
    if mean_sd - mean_c > 0.05:
        print("VERDICT: substrate's decay BEATS counts in non-stationary world.")
        print("This is a real architectural advantage. Decay is a feature, not a bug,")
        print("for environments where old information becomes stale.")
    elif mean_sd - mean_c > -0.05:
        print("VERDICT: substrate ties counts. Decay isn't an advantage here yet.")
    else:
        print("VERDICT: substrate still loses to counts even in non-stationary world.")
        print("The decay rate may be wrong for this drift period, OR counts are")
        print("robust enough that simply having more data outweighs staleness.")


def main():
    question_1_param_sweep()
    question_2_nonstationary()
    print()
    print("=" * 72)
    print("v4 done. See output above.")
    print("=" * 72)


if __name__ == "__main__":
    main()
