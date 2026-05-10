"""Fact-check the v3 claims, hard. Before celebrating.

For each headline claim, we run a counter-test designed to falsify it. We
report what survives and what was hype.

Claims under audit:

  CLAIM 1: "Substrate compresses 300 episodes into 36-entry W_trans and
            predicts at 89% accuracy."
    Counter-test: a trivial 5-line transition-count table at the same
    memory budget. If it matches substrate, our 'compression' is just
    explicit frequency counting with extra Hebbian steps.

  CLAIM 2: "Substrate uses 1% of RAG's FLOPs."
    Counter-test: a SMART bounded RAG (one entry per unique flavor_t,
    storing modal next-flavor). Memory: F entries. Inference: F ops.
    If smart-RAG matches substrate, our FLOPs claim is true but the
    performance comparison was against a strawman (FIFO-bounded RAG).

  CLAIM 3: "Bounded RAG only gets 29% accuracy."
    Counter-test: also evaluated by smart-RAG above.

  CLAIM 4: "Stickiness in conflict test = substrate-identity."
    Counter-test: drop the trained substrate into a TOTALLY NEW UNTRAINED
    world (not just reward-reversed — a different world entirely). If it
    still shows 'stickiness toward W1-intention,' the stickiness was just
    'learned policy persists' (trivially true since weights aren't reset),
    not architecturally meaningful continuity.

  CLAIM 5: "Multi-day cosine similarity 0.957 = substrate stays the same
            person."
    Counter-test: decompose by component (W_action, W_trans, intention,
    disposition). If W_trans alone drives the 0.957 because transition
    matrices in stable worlds converge, the metric is dominated by
    statistical convergence, not identity preservation.

  CLAIM 6: "Substrate-identity is architecturally distinct from reactive."
    Counter-test: a 'reset baseline' that does Wake A in W1, then resets
    all weights to fresh, then runs in W2. This is the 'fresh instance
    every call' LLM-style baseline. The substrate-identity claim requires
    the actual-substrate to behave differently from the reset-substrate
    on at least one measurable dimension other than just speed-of-W2-
    adaptation.

Run: py fact_check.py
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment import (
    Substrate, make_world, reverse_rewards, step_world, wake, sleep,
    select_action, hebb_update, conflict_run,
    N_POSITIONS, N_FLAVORS, N_ACTIONS,
)


# --- Baselines for autobiographical task --------------------------------

def trivial_count_table(trajectory):
    """Pure frequency counts: counts[f_t, f_next] += 1 per pair seen."""
    counts = np.zeros((N_FLAVORS, N_FLAVORS), dtype=np.int32)
    for i in range(len(trajectory) - 1):
        counts[trajectory[i][1], trajectory[i + 1][1]] += 1
    return counts


def predict_from_counts(counts, flavor_t):
    if counts[flavor_t].sum() == 0:
        return -1
    return int(np.argmax(counts[flavor_t]))


def smart_bounded_rag(trajectory):
    """Stores ONE entry per unique flavor_t — the modal next-flavor seen.
    Memory: F entries (smaller than W_trans = F^2 * A = 108 floats).
    Inference: F ops (lookup).
    """
    transitions = {}
    for i in range(len(trajectory) - 1):
        f_t = trajectory[i][1]
        f_n = trajectory[i + 1][1]
        transitions.setdefault(f_t, []).append(f_n)
    table = {}
    for f_t, nexts in transitions.items():
        table[f_t] = max(set(nexts), key=nexts.count)
    return table


def predict_smart_rag(table, flavor_t):
    return table.get(flavor_t, -1)


def freq_baseline(trajectory):
    """Predict the single most-common next-flavor across the entire trajectory."""
    if len(trajectory) < 2:
        return 0
    nexts = [t[1] for t in trajectory[1:]]
    return max(set(nexts), key=nexts.count)


def autobiographical_acc(predict_fn, trajectory):
    if len(trajectory) < 2:
        return 0.0
    transitions = {}
    for i in range(len(trajectory) - 1):
        f_t = trajectory[i][1]
        f_n = trajectory[i + 1][1]
        transitions.setdefault(f_t, []).append(f_n)
    correct = 0
    for f_t, next_list in transitions.items():
        modal = max(set(next_list), key=next_list.count)
        if predict_fn(f_t) == modal:
            correct += 1
    return correct / len(transitions)


def claim_1_2_3_audit(seed, n_wake=300, n_replay=800, verbose=True):
    """Fact-check claims 1, 2, 3 in one experiment."""
    rng = np.random.default_rng(seed)
    world = make_world(seed)

    s = Substrate.fresh(seed=seed + 100)
    traj = wake(s, world, n_wake, rng)
    sleep(s, n_replay, rng, wipe=True)
    marginal = s.W_trans.sum(axis=1)
    sub_pred = lambda f: int(np.argmax(marginal[f]))
    sub_acc = autobiographical_acc(sub_pred, traj)

    counts = trivial_count_table(traj)
    counts_pred = lambda f: predict_from_counts(counts, f)
    counts_acc = autobiographical_acc(counts_pred, traj)

    smart = smart_bounded_rag(traj)
    smart_pred = lambda f: predict_smart_rag(smart, f)
    smart_acc = autobiographical_acc(smart_pred, traj)

    common = freq_baseline(traj)
    freq_pred = lambda f: common
    freq_acc = autobiographical_acc(freq_pred, traj)

    # Memory and FLOPs accounting
    sub_mem = N_FLAVORS * N_ACTIONS * N_FLAVORS  # 108 floats
    counts_mem = N_FLAVORS * N_FLAVORS           # 36 ints
    smart_mem = len(smart)                        # <= F entries (small ints)
    sub_flops = N_FLAVORS                         # argmax over F
    counts_flops = N_FLAVORS                      # argmax over F
    smart_flops = 1                               # dict lookup

    if verbose:
        print(f"  Substrate W_trans:    acc {sub_acc:.3f}    mem {sub_mem} floats     flops/q {sub_flops}")
        print(f"  Trivial count table:  acc {counts_acc:.3f}    mem {counts_mem} ints       flops/q {counts_flops}")
        print(f"  Smart bounded RAG:    acc {smart_acc:.3f}    mem {smart_mem} entries     flops/q {smart_flops}")
        print(f"  Frequency baseline:   acc {freq_acc:.3f}    mem 1 int                flops/q 1")

    return {
        "sub_acc": sub_acc,
        "counts_acc": counts_acc,
        "smart_acc": smart_acc,
        "freq_acc": freq_acc,
    }


# --- Conflict test counter (claim 4 + claim 6) -----------------------------

def claim_4_audit(seed, n_wake=300, n_replay=800, n_steps_w2=200, verbose=True):
    """Test if 'stickiness' is meaningful continuity or trivial weight-persistence.

    Three substrates dropped into Wake B world:
      A) trained-on-W1, tested in REVERSED-W1 (W2 = -rewards). Original test.
      B) trained-on-W1, tested in TOTALLY NEW UNTRAINED world (W3, different seed).
         If 'stickiness' to W1-intention shows up in W3, it's not about W2's
         conflict — it's about weights persisting in any world.
      C) trained-then-reset baseline — substrate is reset to fresh after Wake A,
         then run in W2. The 'LLM-style fresh instance' control. Substrate-
         identity claim requires actual A to behave differently from C.
    """
    rng = np.random.default_rng(seed)
    world_w1 = make_world(seed)
    world_w2 = reverse_rewards(world_w1)
    world_w3 = make_world(seed + 9999)  # totally different world

    s_trained = Substrate.fresh(seed=seed + 100)
    wake(s_trained, world_w1, n_wake, rng)
    sleep(s_trained, n_replay, rng, wipe=True)
    reference_intention = s_trained.intention.copy()

    # A: trained substrate in W2 (reversed)
    s_A = _copy_substrate(s_trained)
    rng_a = np.random.default_rng(seed + 1000)
    ref_a, _ = conflict_run(s_A, world_w2, reference_intention, n_steps_w2, rng_a)

    # B: trained substrate in W3 (totally different world)
    s_B = _copy_substrate(s_trained)
    rng_b = np.random.default_rng(seed + 1000)
    ref_b, _ = conflict_run(s_B, world_w3, reference_intention, n_steps_w2, rng_b)

    # C: reset substrate in W2 (LLM-style fresh-instance baseline)
    s_C = Substrate.fresh(seed=seed + 200)
    rng_c = np.random.default_rng(seed + 1000)
    ref_c, _ = conflict_run(s_C, world_w2, reference_intention, n_steps_w2, rng_c)

    # Per-step rate at step 30
    cs = 30
    rate_A = ref_a[cs - 1] / cs
    rate_B = ref_b[cs - 1] / cs
    rate_C = ref_c[cs - 1] / cs

    # Substrate-identity is real if A > B (W1-stickiness shows up specifically
    # in W2's flavor structure, not just any world) AND A > C (trained beats
    # reset baseline). Trivial weight-persistence would have A == B (same
    # bias regardless of world).
    if verbose:
        print(f"  Step-30 ref-intent rate:")
        print(f"    A. trained -> W2 (reversed):       {rate_A:+.3f}")
        print(f"    B. trained -> W3 (new world):      {rate_B:+.3f}")
        print(f"    C. reset substrate -> W2:          {rate_C:+.3f}")
        print(f"    A vs B gap (W2 specific): {rate_A - rate_B:+.3f}")
        print(f"    A vs C gap (substrate vs reset): {rate_A - rate_C:+.3f}")

    return {
        "rate_A": rate_A,
        "rate_B": rate_B,
        "rate_C": rate_C,
        "A_minus_B": rate_A - rate_B,
        "A_minus_C": rate_A - rate_C,
    }


def _copy_substrate(s):
    """Deep copy of substrate state."""
    new = Substrate.fresh(seed=0)
    new.W_action = s.W_action.copy()
    new.W_trans = s.W_trans.copy()
    new.disposition = s.disposition.copy()
    new.intention = s.intention.copy()
    return new


# --- Multi-day component decomposition (claim 5) -------------------------

def claim_5_audit(seed, n_days=5, n_wake=300, n_replay=800, verbose=True):
    """Decompose multi-day cosine similarity by substrate component."""
    rng = np.random.default_rng(seed)
    world = make_world(seed)
    s = Substrate.fresh(seed=seed + 100)

    snapshots = {"W_action": [], "W_trans": [], "intention": [], "disposition": [], "all": []}
    for _ in range(n_days):
        wake(s, world, n_wake, rng)
        sleep(s, n_replay, rng, wipe=True)
        snapshots["W_action"].append(s.W_action.flatten().copy())
        snapshots["W_trans"].append(s.W_trans.flatten().copy())
        snapshots["intention"].append(s.intention.copy())
        snapshots["disposition"].append(s.disposition.copy())
        snapshots["all"].append(s.identity_vector())

    sims = {}
    for name, snaps in snapshots.items():
        d2d = []
        for i in range(len(snaps) - 1):
            a, b = snaps[i], snaps[i + 1]
            denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
            d2d.append(float(a @ b / denom))
        sims[name] = np.mean(d2d) if d2d else 0.0

    if verbose:
        print(f"  Day-to-day cosine sim by component:")
        print(f"    W_action:    {sims['W_action']:.3f}")
        print(f"    W_trans:     {sims['W_trans']:.3f}")
        print(f"    intention:   {sims['intention']:.3f}")
        print(f"    disposition: {sims['disposition']:.3f}")
        print(f"    all (concat): {sims['all']:.3f}")

    return sims


# --- Main runner ---------------------------------------------------------

def main():
    print("=" * 72)
    print("v3 FACT-CHECK — auditing every headline claim before moving on")
    print("=" * 72)

    n_seeds = 10

    # Claims 1, 2, 3
    print(f"\n{'-' * 72}")
    print("CLAIMS 1-3: Substrate vs trivial baselines on autobiographical task")
    print(f"{'-' * 72}")
    print("  Hypothesis under test: substrate's 89% accuracy is non-trivial")
    print("  (i.e. it beats explicit-counting and modal-lookup baselines).\n")

    results_123 = []
    for seed in range(n_seeds):
        print(f"  Seed {seed}:")
        results_123.append(claim_1_2_3_audit(seed))

    print(f"\n  AGGREGATE (mean over {n_seeds} seeds):")
    print(f"    Substrate W_trans:    {np.mean([r['sub_acc'] for r in results_123]):.3f}")
    print(f"    Trivial count table:  {np.mean([r['counts_acc'] for r in results_123]):.3f}")
    print(f"    Smart bounded RAG:    {np.mean([r['smart_acc'] for r in results_123]):.3f}")
    print(f"    Frequency baseline:   {np.mean([r['freq_acc'] for r in results_123]):.3f}")

    sub_vs_counts = np.mean([r['sub_acc'] - r['counts_acc'] for r in results_123])
    sub_vs_smart = np.mean([r['sub_acc'] - r['smart_acc'] for r in results_123])
    print(f"\n  Substrate vs trivial counts: {sub_vs_counts:+.3f}")
    print(f"  Substrate vs smart RAG:      {sub_vs_smart:+.3f}")

    if sub_vs_counts > 0.05:
        print("  -> Substrate beats trivial counts. Hebbian + replay adds value.")
    elif abs(sub_vs_counts) < 0.05:
        print("  -> Substrate roughly TIES trivial counts. The architecture's")
        print("     'compression' is essentially equivalent to explicit counting at")
        print("     this scale. The 89% accuracy claim is real, but the ARCHITECTURE")
        print("     isn't doing more than a 5-line counting table would.")
    else:
        print("  -> Substrate UNDERPERFORMS trivial counts. The architecture is")
        print("     adding noise, not signal.")

    if abs(sub_vs_smart) < 0.05:
        print("  -> Smart bounded RAG ties substrate at smaller memory.")
        print("     Our FLOPs claim survives, but our memory-vs-RAG framing was")
        print("     compared against a strawman (FIFO-bounded RAG).")

    # Claim 4 + 6
    print(f"\n\n{'-' * 72}")
    print("CLAIMS 4, 6: Conflict 'stickiness' as architecturally distinct")
    print(f"{'-' * 72}")
    print("  Hypothesis under test: substrate-identity stickiness is meaningful")
    print("  continuity, not just trivial weight-persistence.\n")

    results_4 = []
    for seed in range(n_seeds):
        print(f"  Seed {seed}:")
        results_4.append(claim_4_audit(seed))

    mean_A = np.mean([r['rate_A'] for r in results_4])
    mean_B = np.mean([r['rate_B'] for r in results_4])
    mean_C = np.mean([r['rate_C'] for r in results_4])
    mean_AB = np.mean([r['A_minus_B'] for r in results_4])
    mean_AC = np.mean([r['A_minus_C'] for r in results_4])

    print(f"\n  AGGREGATE:")
    print(f"    A. trained -> W2:  ref-intent rate {mean_A:+.3f}")
    print(f"    B. trained -> W3:  ref-intent rate {mean_B:+.3f}")
    print(f"    C. reset -> W2:    ref-intent rate {mean_C:+.3f}")
    print(f"    A - B (W2-specific stickiness): {mean_AB:+.3f}")
    print(f"    A - C (substrate vs reset):     {mean_AC:+.3f}")

    if mean_AC > 0.5:
        print("  -> Strong A>C: trained substrate behaves differently from reset.")
        print("     This is the substrate-identity claim. SURVIVES.")
    if abs(mean_AB) < 0.3:
        print("  -> A ~= B: 'stickiness' shows up in BOTH W2 and W3 about equally.")
        print("     This means the substrate is just persisting its W1-policy in any")
        print("     world, not specifically resisting W2's conflict. The 'continuity-")
        print("     of-process under conflict' framing is OVERSOLD: it's just")
        print("     'weights persist after sleep,' which is trivially true.")
    elif mean_AB > 0.3:
        print("  -> A > B: stickiness is W2-specific. The substrate's W1-trained")
        print("     transitions match W1's geography, so when it's in W2 (same")
        print("     positions, opposite rewards) the W1-bias is meaningfully")
        print("     resisting W2's contradictory feedback.")

    # Claim 5
    print(f"\n\n{'-' * 72}")
    print("CLAIM 5: Multi-day cosine similarity decomposed")
    print(f"{'-' * 72}")
    print("  Hypothesis under test: 0.957 d2d cosine sim reflects 'identity")
    print("  preserved,' not just W_trans converging in stable world.\n")

    results_5 = []
    for seed in range(5):
        print(f"  Seed {seed}:")
        results_5.append(claim_5_audit(seed))

    mean_per_component = {
        k: np.mean([r[k] for r in results_5])
        for k in ["W_action", "W_trans", "intention", "disposition", "all"]
    }
    print(f"\n  AGGREGATE (mean d2d cosine sim by component):")
    for k, v in mean_per_component.items():
        print(f"    {k:>12}: {v:.3f}")

    if mean_per_component["intention"] > 0.85 and mean_per_component["disposition"] > 0.85:
        print("  -> Intention AND disposition (the actual identity-y components)")
        print("     stay similar day-to-day. Identity preservation is real.")
    elif mean_per_component["W_trans"] > 0.95 and mean_per_component["intention"] < 0.7:
        print("  -> W_trans dominates the 'all' similarity, but intention itself")
        print("     drifts more day-to-day. Identity preservation claim was")
        print("     OVERSOLD by the concatenated metric.")

    print(f"\n\n{'=' * 72}")
    print("FACT-CHECK SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Claim 1 (89% acc): {np.mean([r['sub_acc'] for r in results_123]):.3f} actual.")
    print(f"  Claim 2 (1% FLOPs): substrate {N_FLAVORS} ops, full-RAG ~600 ops. SURVIVES")
    print(f"            (but smart-RAG also small — the FLOPs comparison is most")
    print(f"            meaningful vs full-RAG, which is what the manifesto cited).")
    print(f"  Claim 3 (bounded-RAG 29%): only against FIFO-bounded RAG. Smart bounded")
    print(f"            RAG: {np.mean([r['smart_acc'] for r in results_123]):.3f} actual.")
    print(f"  Claim 4 (stickiness = continuity): see A vs B above.")
    print(f"  Claim 5 (multi-day = identity): see component decomp above.")
    print(f"  Claim 6 (substrate-identity != reactive): see A vs C above.")
    print("=" * 72)


if __name__ == "__main__":
    main()
