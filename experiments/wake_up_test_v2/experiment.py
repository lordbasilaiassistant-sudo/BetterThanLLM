"""Wake-Up Test v2 — substrate-identity falsifier with intention component.

Adds over v1:
  - W_trans: (flavor, action, next_flavor) full transition tensor
  - intention vector: target-flavor preference, persists across the gap
  - goal-directed planning: W_trans + intention biases action selection
  - real continuity test (steps to reach highest-intention flavor)

The v1 substrate is a strict subset of v2 (zero intention + plan_weight=0).

Pure numpy. Local Hebbian. No backprop. No anterograde scaffolding.

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
    """Generate a world with at least one strongly-positive flavor.

    Random rewards in [-1, 1] sometimes produce all-negative or all-flat
    worlds where no agent can develop a meaningful goal. We guarantee at
    least one flavor has reward >= 0.5 so intention has a target to form.
    """
    rng = np.random.default_rng(seed)
    pos_to_flavor = rng.integers(0, N_FLAVORS, size=N_POSITIONS)
    flavor_to_reward = rng.uniform(-1, 1, size=N_FLAVORS)
    if flavor_to_reward.max() < 0.5:
        flavor_to_reward[int(np.argmax(flavor_to_reward))] = float(rng.uniform(0.5, 1.0))
    return pos_to_flavor, flavor_to_reward


def step_world(pos, action, n_positions=N_POSITIONS):
    if action == 0:
        return max(0, pos - 1)
    if action == 1:
        return min(n_positions - 1, pos + 1)
    return pos


# --- Substrate -----------------------------------------------------------

@dataclass
class Substrate:
    W_action: np.ndarray         # (F, A)         flavor -> action preference
    W_trans: np.ndarray          # (F, A, F)      flavor x action -> next-flavor expectation
    disposition: np.ndarray      # (F,)           rolling reward avg per flavor (water layer)
    intention: np.ndarray        # (F,)           current-goal preference; persists across gap
    episodic: List[Tuple[int, int, float, int]] = field(default_factory=list)
    lr: float = 0.05
    disp_lr: float = 0.02
    intent_lr: float = 0.04
    plan_weight: float = 0.6
    top_k: int = 2

    @classmethod
    def fresh(cls, seed, intent_lr=0.04, plan_weight=0.6):
        rng = np.random.default_rng(seed)
        return cls(
            W_action=rng.normal(0, 0.01, (N_FLAVORS, N_ACTIONS)),
            W_trans=rng.normal(0, 0.01, (N_FLAVORS, N_ACTIONS, N_FLAVORS)),
            disposition=np.zeros(N_FLAVORS),
            intention=np.zeros(N_FLAVORS),
            intent_lr=intent_lr,
            plan_weight=plan_weight,
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
    """Reactive policy + goal-directed planning + water-layer disposition bias."""
    react_logits = s.W_action[flavor].copy()
    # Disposition (water layer): bias STAY when current flavor's avg reward is positive
    react_logits[2] += 0.3 * s.disposition[flavor]
    # Planning: for each action, expected intention value at predicted next-flavor
    plan_logits = np.zeros(N_ACTIONS)
    for a in range(N_ACTIONS):
        next_dist = softmax(s.W_trans[flavor, a])
        plan_logits[a] = next_dist @ s.intention
    logits = react_logits + s.plan_weight * plan_logits
    probs = softmax_topk(logits, s.top_k)
    return int(rng.choice(N_ACTIONS, p=probs)), probs


def hebb_update(s, flavor, action, reward, next_flavor, allow_intent=True):
    s.W_action[flavor, action] += s.lr * reward
    s.W_action[flavor] *= (1 - s.lr * 0.05)
    s.W_trans[flavor, action, next_flavor] += s.lr
    s.W_trans[flavor, action] *= (1 - s.lr * 0.05)
    s.disposition[flavor] += s.disp_lr * (reward - s.disposition[flavor])
    if allow_intent:
        # Intention forms toward flavors that yield BETTER-THAN-AVERAGE rewards,
        # not just absolute-positive. Robust across worlds with all-positive or
        # all-negative reward distributions.
        baseline = float(s.disposition.mean())
        relative = reward - baseline
        if relative > 0.05:
            s.intention[next_flavor] += s.intent_lr * relative
            s.intention *= (1 - s.intent_lr * 0.05)


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
    """Predict next-flavor from W_trans marginal (sum over actions).

    Score per UNIQUE flavor_t (modal next-flavor in trajectory) so a
    repetitive trajectory with one heavy transition can't inflate the
    metric and let a random control look strong.
    """
    if len(trajectory) < 2:
        return 0.0
    # Build empirical modal next-flavor per flavor_t
    transitions = {}
    for i in range(len(trajectory) - 1):
        f_t = trajectory[i][1]
        f_n = trajectory[i + 1][1]
        transitions.setdefault(f_t, []).append(f_n)
    if not transitions:
        return 0.0
    marginal = s.W_trans.sum(axis=1)
    correct = 0
    for f_t, next_list in transitions.items():
        modal = max(set(next_list), key=next_list.count)
        if int(np.argmax(marginal[f_t])) == modal:
            correct += 1
    return correct / len(transitions)


def test_continuity_score(s, world, reference_intention, n_steps, rng):
    """Cumulative reference-intention value over a long horizon.

    Both trained and control are scored against the SAME reference
    (trained substrate's intention vector). The question this answers:
    does the substrate's policy pursue this intention? Trained has the
    intention internally and uses W_trans to plan; control has zero
    intention -> no planning -> wanders. If trained accumulates more
    reference-intention value per step, planning is real.

    Robust to diffuse intention (rewards all positive-intention flavors,
    not just argmax) and to start-position luck (long horizon averages
    out trivial wins).
    """
    pos_to_flavor, _ = world
    pos = int(rng.integers(0, N_POSITIONS))
    score = 0.0
    for _ in range(n_steps):
        flavor = int(pos_to_flavor[pos])
        score += float(reference_intention[flavor])
        action, _ = select_action(s, flavor, rng)
        pos = step_world(pos, action)
    return score / max(n_steps, 1)


def test_continuity_to_target(s, world, target_flavor, n_steps, rng, start_pos=0):
    """Steps to first reach a SPECIFIC target_flavor from fixed start.

    Used to measure both the trained agent reaching its own intention target
    and the control walking toward the same target (without its own goal).
    """
    pos_to_flavor, _ = world
    pos = start_pos
    for step in range(n_steps):
        if int(pos_to_flavor[pos]) == target_flavor:
            return step
        flavor = int(pos_to_flavor[pos])
        action, _ = select_action(s, flavor, rng)
        pos = step_world(pos, action)
    return n_steps


# --- Protocol ------------------------------------------------------------

def run_one_seed(seed, n_wake=300, n_replay=800, n_test=25, verbose=True):
    rng = np.random.default_rng(seed)
    world = make_world(seed)
    pos_to_flavor, flavor_to_reward = world
    gt_target = int(np.argmax(flavor_to_reward))

    s_trained = Substrate.fresh(seed=seed + 100)
    if verbose:
        print(f"[Wake A] {n_wake} steps...")
    traj = wake(s_trained, world, n_steps=n_wake, rng=rng)
    intent_pre_sleep = s_trained.intention.copy()
    if verbose:
        print(f"[Sleep] replay {n_replay}x; wipe buffer.")
    sleep(s_trained, n_replay=n_replay, rng=rng, wipe=True)
    intent_post_sleep = s_trained.intention.copy()
    intent_drift = float(np.linalg.norm(intent_post_sleep - intent_pre_sleep))
    intent_max = float(s_trained.intention.max())
    intent_aligns_gt = (int(np.argmax(s_trained.intention)) == gt_target) if intent_max > 0.05 else False
    assert len(s_trained.episodic) == 0, "Episodic buffer must be wiped"

    s_control = Substrate.fresh(seed=seed + 200)

    # Negative controls — same trained substrate with one component zeroed,
    # to attribute continuity to specific architectural pieces
    s_no_intent = Substrate.fresh(seed=seed + 100)
    s_no_intent.W_action = s_trained.W_action.copy()
    s_no_intent.W_trans = s_trained.W_trans.copy()
    s_no_intent.disposition = s_trained.disposition.copy()
    s_no_intent.intention = np.zeros_like(s_trained.intention)

    s_no_trans = Substrate.fresh(seed=seed + 100)
    s_no_trans.W_action = s_trained.W_action.copy()
    s_no_trans.W_trans = np.zeros_like(s_trained.W_trans)
    s_no_trans.disposition = s_trained.disposition.copy()
    s_no_trans.intention = s_trained.intention.copy()

    if verbose:
        print("[Wake B] Probing...")
    sm_t = test_self_model(s_trained, world, n_trials=1000, rng=np.random.default_rng(seed + 1))
    sm_c = test_self_model(s_control, world, n_trials=1000, rng=np.random.default_rng(seed + 1))
    auto_t = test_autobiographical(s_trained, traj)
    auto_c = test_autobiographical(s_control, traj)

    # Continuity: cumulative trained-intention score over long horizon.
    # All variants scored against the SAME reference (trained substrate's
    # own intention) — only their POLICY differs.
    ref_intent = s_trained.intention.copy()
    cont_horizon = 100
    cont_t = test_continuity_score(s_trained, world, ref_intent, cont_horizon, np.random.default_rng(seed + 3))
    cont_c = test_continuity_score(s_control, world, ref_intent, cont_horizon, np.random.default_rng(seed + 3))
    cont_no_intent = test_continuity_score(s_no_intent, world, ref_intent, cont_horizon, np.random.default_rng(seed + 3))
    cont_no_trans = test_continuity_score(s_no_trans, world, ref_intent, cont_horizon, np.random.default_rng(seed + 3))

    pass_self = (sm_t - sm_c) > 0.05
    pass_auto = (auto_t - auto_c) > 0.10
    pass_cont = (cont_t - cont_c) > 0.20

    if verbose:
        print(f"  Self-model        | trained {sm_t:+.3f}   control {sm_c:+.3f}   delta {sm_t - sm_c:+.3f}   {'PASS' if pass_self else 'fail'}")
        print(f"  Autobiographical  | trained {auto_t:.3f}    control {auto_c:.3f}    delta {auto_t - auto_c:+.3f}   {'PASS' if pass_auto else 'fail'}")
        print(f"  Continuity (intent score per step) | trained {cont_t:.3f}   control {cont_c:.3f}   delta {cont_t - cont_c:+.3f}   {'PASS' if pass_cont else 'fail'}")
        print(f"    no-intent ablation: {cont_no_intent:.3f}   no-W_trans ablation: {cont_no_trans:.3f}")
        print(f"  Intention | drift {intent_drift:.2f}   max {intent_max:.2f}   aligns-ground-truth: {intent_aligns_gt}")

    return {
        "self_model": (sm_t, sm_c, pass_self),
        "autobiographical": (auto_t, auto_c, pass_auto),
        "continuity": (cont_t, cont_c, pass_cont),
        "ablations": {"no_intent": cont_no_intent, "no_trans": cont_no_trans},
        "intent_drift": intent_drift,
        "intent_max": intent_max,
        "intent_aligns_gt": intent_aligns_gt,
        "pass": pass_self and pass_auto and pass_cont,
    }


def main():
    print("=" * 60)
    print("Wake-Up Test v2 — substrate-identity + intention")
    print("=" * 60)

    results = []
    for seed in range(10):
        print(f"\n--- Seed {seed} ---")
        results.append(run_one_seed(seed))

    print(f"\n\n{'=' * 60}\nSUMMARY (10 seeds)\n{'=' * 60}")
    n_pass = sum(r['pass'] for r in results)
    n_self = sum(r['self_model'][2] for r in results)
    n_auto = sum(r['autobiographical'][2] for r in results)
    n_cont = sum(r['continuity'][2] for r in results)
    print(f"Full pass (all 3):       {n_pass}/10")
    print(f"Self-model individual:   {n_self}/10")
    print(f"Autobiographical:        {n_auto}/10")
    print(f"Continuity:              {n_cont}/10")

    sm_d = [r['self_model'][0] - r['self_model'][1] for r in results]
    auto_d = [r['autobiographical'][0] - r['autobiographical'][1] for r in results]
    cont_d = [r['continuity'][0] - r['continuity'][1] for r in results]
    drift = [r['intent_drift'] for r in results]
    aligns = [r['intent_aligns_gt'] for r in results]
    abl_no_intent = [r['ablations']['no_intent'] for r in results]
    abl_no_trans = [r['ablations']['no_trans'] for r in results]
    cont_trained = [r['continuity'][0] for r in results]
    cont_control = [r['continuity'][1] for r in results]

    print(f"\nMean deltas (trained vs control):")
    print(f"  Self-model:                 {np.mean(sm_d):+.3f}  +- {np.std(sm_d):.3f}")
    print(f"  Autobiographical:           {np.mean(auto_d):+.3f}  +- {np.std(auto_d):.3f}")
    print(f"  Continuity (intent score):  {np.mean(cont_d):+.3f}  +- {np.std(cont_d):.3f}")
    print(f"  Intention drift over sleep: {np.mean(drift):.3f}  +- {np.std(drift):.3f}")
    print(f"  Intention aligns w/ ground truth: {sum(aligns)}/10")
    print(f"\nAblations — mean cumulative reference-intent score per step (higher = pursued goal):")
    print(f"  Trained (full substrate):    {np.mean(cont_trained):.3f}")
    print(f"  Control (zero substrate):    {np.mean(cont_control):.3f}")
    print(f"  No intention (zeroed):       {np.mean(abl_no_intent):.3f}")
    print(f"  No W_trans (zeroed):         {np.mean(abl_no_trans):.3f}")

    print(f"\n{'=' * 60}\nVERDICT\n{'=' * 60}")
    if n_pass >= 8:
        print("v2 PASSED. All three reconstitutions hold across 10 seeds.")
        print("Substrate-identity thesis validated at full-system level.")
    elif n_pass >= 6:
        print("v2 PASSED with margin. Strong evidence; one or two edge seeds.")
    elif n_self >= 8 and n_auto >= 8:
        print("v2 partial. Self-model + autobiographical solid; continuity weak.")
    else:
        print("v2 FAILED. Re-examine architecture or test design.")
    print("=" * 60)


if __name__ == "__main__":
    main()
