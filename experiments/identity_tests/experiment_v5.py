"""Identity Tests v5 — cross-world identity & component ablation.

The most important remaining test: is identity bound to substrate or to
environment? Take a substrate trained in W1, drop it into W2 (totally
different world). Does it stay closer to its W1-self or become someone new?

Plus: which substrate component carries identity most? Ablation by zeroing
each component and seeing how much signature changes.

Run: py experiment_v5.py
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_v4 import (
    Substrate, make_world, step_world,
    select_action, hebb_update, wake, sleep,
    behavioral_signature, signature_similarity,
    N_POSITIONS, N_FLAVORS, N_ACTIONS,
)


# --- T8: Cross-world identity preservation ------------------------------

def t8_cross_world(seed, n_wake_w1=500, n_replay_w1=1000,
                   n_wake_w2=500, n_replay_w2=1000):
    """Substrate trained in W1, then fully run in W2. Compare:
      (a) new signature vs original W1 self-signature (does identity persist?)
      (b) new signature vs a fresh substrate trained from scratch in W2

    If (a) > (b), identity persists across world change. If (b) > (a),
    environment overwrote identity.
    """
    rng = np.random.default_rng(seed)
    world_w1 = make_world(seed)
    world_w2 = make_world(seed + 9999)  # totally different world

    # Train original substrate in W1, capture W1-self-signature
    s_orig = Substrate.fresh(seed=seed + 100)
    wake(s_orig, world_w1, n_wake_w1, rng)
    sleep(s_orig, n_replay_w1, rng, wipe=True)
    sig_w1_self = behavioral_signature(s_orig, rng=np.random.default_rng(seed + 333))

    # Same substrate now lives in W2
    s_traveled = s_orig.deep_clone()
    rng_w2 = np.random.default_rng(seed + 7777)
    wake(s_traveled, world_w2, n_wake_w2, rng_w2)
    sleep(s_traveled, n_replay_w2, rng_w2, wipe=True)
    sig_after_w2 = behavioral_signature(s_traveled, rng=np.random.default_rng(seed + 333))

    # Fresh substrate trained from scratch in W2
    s_fresh_w2 = Substrate.fresh(seed=seed + 200)
    rng_fresh = np.random.default_rng(seed + 7777)
    wake(s_fresh_w2, world_w2, n_wake_w2, rng_fresh)
    sleep(s_fresh_w2, n_replay_w2, rng_fresh, wipe=True)
    sig_fresh_w2 = behavioral_signature(s_fresh_w2, rng=np.random.default_rng(seed + 333))

    # Comparisons
    a = signature_similarity(sig_after_w2, sig_w1_self)      # post-W2 vs W1-self
    b = signature_similarity(sig_after_w2, sig_fresh_w2)     # post-W2 vs fresh-in-W2

    return {
        "post_w2_vs_w1_self": a,
        "post_w2_vs_fresh_in_w2": b,
        "identity_signal": a - b,  # positive = identity persisted across world change
    }


# --- T9: Component ablation ---------------------------------------------

def t9_component_ablation(seed, n_wake=500, n_replay=1000):
    """Which substrate component carries identity? Train substrate. Capture
    signature. Then ablate (zero out) each component separately and measure
    how much signature changes. Bigger drop = more important to identity.
    """
    rng = np.random.default_rng(seed)
    world = make_world(seed)
    s = Substrate.fresh(seed=seed + 100)
    wake(s, world, n_wake, rng)
    sleep(s, n_replay, rng, wipe=True)
    sig_full = behavioral_signature(s, rng=np.random.default_rng(seed + 333))

    results = {}
    for component in ["W_action", "W_trans", "intention", "disposition"]:
        s_abl = s.deep_clone()
        if component == "W_action":
            s_abl.W_action = np.zeros_like(s_abl.W_action)
        elif component == "W_trans":
            s_abl.W_trans = np.zeros_like(s_abl.W_trans)
        elif component == "intention":
            s_abl.intention = np.zeros_like(s_abl.intention)
        elif component == "disposition":
            s_abl.disposition = np.zeros_like(s_abl.disposition)
        sig_abl = behavioral_signature(s_abl, rng=np.random.default_rng(seed + 333))
        sim = signature_similarity(sig_full, sig_abl)
        results[component] = sim
    return results


# --- T10: Long-horizon identity (50 cycles) -----------------------------

def t10_long_horizon(seed, n_cycles=50, n_wake=200, n_replay=400):
    """Does identity hold across 50 sleep cycles? (5x the v4 test.)"""
    rng = np.random.default_rng(seed)
    world = make_world(seed)
    s = Substrate.fresh(seed=seed + 100)

    # Capture baseline at cycle 5 (after some warm-up)
    for _ in range(5):
        wake(s, world, n_wake, rng)
        sleep(s, n_replay, rng, wipe=True)
    baseline_sig = behavioral_signature(s, rng=np.random.default_rng(seed + 333))

    sims = []
    for cycle in range(n_cycles - 5):
        wake(s, world, n_wake, rng)
        sleep(s, n_replay, rng, wipe=True)
        sig = behavioral_signature(s, rng=np.random.default_rng(seed + 333))
        sims.append(signature_similarity(baseline_sig, sig))

    return {
        "sims_to_baseline": sims,
        "final": sims[-1],
        "min": min(sims),
        "mean": float(np.mean(sims)),
    }


# --- Main ----------------------------------------------------------------

def main():
    print("=" * 72)
    print("IDENTITY TESTS v5 — cross-world & ablation")
    print("=" * 72)

    print("\n--- T8: Cross-world identity preservation ---")
    print("Substrate trained in W1, fully run in W2. Does identity persist?")
    t8 = [t8_cross_world(s) for s in range(10)]
    for i, r in enumerate(t8):
        sig = "+" if r['identity_signal'] > 0 else "-"
        print(f"  Seed {i}: post-W2 vs W1-self {r['post_w2_vs_w1_self']:.4f} | "
              f"vs fresh-in-W2 {r['post_w2_vs_fresh_in_w2']:.4f} | "
              f"id signal {sig}{abs(r['identity_signal']):.4f}")
    mean_id_signal = np.mean([r['identity_signal'] for r in t8])
    print(f"\n  Mean identity signal: {mean_id_signal:+.4f}")
    if mean_id_signal > 0:
        print("  -> Identity persists across world change. The substrate IS more")
        print("     itself than it is its environment.")
    else:
        print("  -> Environment dominates. Identity overwritten by new world.")

    print("\n--- T9: Component ablation (which carries identity?) ---")
    t9 = [t9_component_ablation(s) for s in range(10)]
    for component in ["W_action", "W_trans", "intention", "disposition"]:
        sims = [r[component] for r in t9]
        # Lower sim = more important component (zeroing it changes signature more)
        print(f"  Zero {component:<12}: mean sim to full {np.mean(sims):.4f}  (lower = more critical)")

    print("\n--- T10: Long-horizon identity (50 sleep cycles) ---")
    t10 = [t10_long_horizon(s) for s in range(3)]
    for i, r in enumerate(t10):
        print(f"  Seed {i}: final-vs-baseline {r['final']:.4f}, min {r['min']:.4f}, mean {r['mean']:.4f}")
    print(f"  Average across seeds: final {np.mean([r['final'] for r in t10]):.4f}, "
          f"min {np.mean([r['min'] for r in t10]):.4f}")

    print(f"\n{'=' * 72}")
    print("v5 SUMMARY")
    print(f"{'=' * 72}")
    print(f"  T8 cross-world identity signal: {mean_id_signal:+.4f}")
    print(f"  T9 component criticality (lower = more critical to identity):")
    for component in ["W_action", "W_trans", "intention", "disposition"]:
        sims = [r[component] for r in t9]
        print(f"     zero {component:<12}: {np.mean(sims):.4f}")
    print(f"  T10 50-cycle identity preservation (mean to baseline): {np.mean([r['mean'] for r in t10]):.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
