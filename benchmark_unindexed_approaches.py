"""
Benchmark the three unindexed scaffolding approaches.

N = 100 (design length), M = 4 (motif residues).
Each approach is timed over multiple iterations after JIT warmup.
"""
import time
from itertools import product
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platforms", "cpu")

from mosaic.util import kabsch

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
N = 100   # design length
M = 4     # motif residues
K = 15    # top-K candidates per motif residue
N_REPEATS = 200

key = jax.random.key(0)
k1, k2, k3, k4 = jax.random.split(key, 4)

pred_ca = jax.random.normal(k1, (N, 3))          # predicted CA coords
gt_ca = jax.random.normal(k2, (M, 3))            # ground-truth motif CA
pssm = jax.nn.softmax(jax.random.normal(k3, (N, 20)), axis=-1)  # design PSSM
gt_aa_indices = jax.random.randint(k4, (M,), 0, 20)             # motif AA ids

gt_distogram = jnp.sqrt(
    jnp.sum((gt_ca[:, None, :] - gt_ca[None, :, :]) ** 2, axis=-1) + 1e-8
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rmsd_at_indices(indices, pred_ca, gt_ca):
    """RMSD after Kabsch alignment for a single candidate assignment."""
    sel = pred_ca[indices]
    R, t = kabsch(sel, gt_ca)
    aligned = sel @ R + t
    return jnp.sqrt(jnp.mean(jnp.sum((aligned - gt_ca) ** 2, axis=-1)))


def timeit(fn, n=N_REPEATS, label=""):
    """Time a function over n calls, return mean ms."""
    fn()  # warmup / compile
    jax.block_until_ready(fn())  # ensure compiled

    t0 = time.perf_counter()
    for _ in range(n):
        result = fn()
        jax.block_until_ready(result)
    elapsed = (time.perf_counter() - t0) / n * 1000
    print(f"  {label}: {elapsed:.3f} ms/call  ({n} calls)")
    return elapsed


# ===========================================================================
# Approach 1: Co-optimized Assignment Matrix
# ===========================================================================

def approach1_soft(assignment_logits, pred_ca, gt_ca, temperature=1.0):
    """
    Soft assignment: weighted average of predicted coords per motif residue,
    then compute RMSD on the blended coordinates.

    assignment_logits: [N, M]  (no +1 column for simplicity in timing)
    """
    weights = jax.nn.softmax(assignment_logits / temperature, axis=0)  # [N, M]
    # weighted coords per motif residue: [M, 3]
    soft_coords = jnp.einsum("nm, nd -> md", weights, pred_ca)
    R, t = kabsch(soft_coords, gt_ca)
    aligned = soft_coords @ R + t
    return jnp.sqrt(jnp.mean(jnp.sum((aligned - gt_ca) ** 2, axis=-1)))


def approach1_topk_enum(assignment_logits, pred_ca, gt_ca, k=K):
    """
    Hard enumeration variant: take top-K positions per motif column,
    enumerate all K^M assignments, evaluate RMSD for each via vmap.
    """
    weights = jax.nn.softmax(assignment_logits, axis=0)  # [N, M]

    # top-K indices per motif residue
    topk_per_motif = []
    for j in range(M):
        _, idx = jax.lax.top_k(weights[:, j], k)
        topk_per_motif.append(idx)

    # Cartesian product of top-K indices → [K^M, M]
    combos = jnp.array(list(product(*[range(k)] * M)))  # index into topk arrays
    candidate_indices = jnp.stack(
        [topk_per_motif[j][combos[:, j]] for j in range(M)], axis=-1
    )  # [n_candidates, M]

    # weight per candidate = product of assignment probs
    candidate_weights = jnp.prod(
        jnp.stack([weights[candidate_indices[:, j], j] for j in range(M)], axis=-1),
        axis=-1,
    )  # [n_candidates]

    # vmap RMSD over all candidates
    rmsds = jax.vmap(rmsd_at_indices, in_axes=(0, None, None))(
        candidate_indices, pred_ca, gt_ca
    )

    # weighted expected RMSD
    return jnp.sum(candidate_weights * rmsds) / jnp.sum(candidate_weights)


# JIT compile
assignment_logits = jax.random.normal(jax.random.key(10), (N, M))

approach1_soft_jit = jax.jit(partial(approach1_soft, pred_ca=pred_ca, gt_ca=gt_ca))
approach1_enum_jit = jax.jit(partial(approach1_topk_enum, pred_ca=pred_ca, gt_ca=gt_ca, k=K))


# ===========================================================================
# Approach 2: Geometry-First (Distance Matching)
# ===========================================================================

def approach2_geometry(pred_ca, gt_ca, gt_distogram, k=K):
    """
    1. Compute pred distance matrix [N, N].
    2. For each motif residue, score all N positions by distance-profile similarity.
    3. Take top-K per motif, enumerate candidates, pick best RMSD.
    """
    # predicted pairwise distances [N, N]
    pred_dist = jnp.sqrt(
        jnp.sum((pred_ca[:, None, :] - pred_ca[None, :, :]) ** 2, axis=-1) + 1e-8
    )

    # For each motif residue j, we want to find which design positions have
    # a distance profile to other positions that matches gt_distogram[j, :].
    # Approximate: for each candidate position i, score = sum over other motif
    # residues of how well its distance to the best-matching position fits.
    # Simpler approach: score each position by average distance to the
    # ground-truth centroid-like criterion. Here we use a pragmatic proxy:
    # for each pair (i, i'), compute |pred_dist[i,i'] - gt_dist[j,j']| and
    # aggregate.

    # Brute force: enumerate all C(N, M) ordered subsets.
    # With N=100, M=4: C(100,4) ~ 3.9M — let's use top-K shortcut instead.

    # Per-motif scoring: for each motif residue j, rank positions by how well
    # they could serve as that residue (using distance profile to all N positions
    # vs the gt distances). We use a simplified score: for motif residue 0,
    # just take top-K by proximity to centroid. For subsequent ones, refine.

    # Practical shortcut: take top-K candidates per position uniformly,
    # enumerate K^M candidates, evaluate RMSD.
    # Score: mean absolute distance-profile error to gt for each position-motif pair.

    # For each design position i and motif residue j:
    # score[i, j] = min over possible assignments of remaining residues of
    # |d(i, .) - gt_d(j, .)|
    # This is expensive, so use a simpler proxy: just use the top-K closest
    # positions to each other (sorted by pairwise compactness matching gt).

    # Simple proxy score: for each position i, for motif residue j,
    # look at the distribution of distances from i to all other positions,
    # and compare to gt_distogram[j, :] in a summary stat.
    # Use: for each (i, j), score = -sum_j' |sorted_dist_from_i[j'] - sorted(gt_distogram[j])[j']|

    # Even simpler and fast: just use all-pairs approach.
    # Take top-K by random (or uniform) per motif and evaluate RMSD.
    # For the benchmark, the key cost is the vmap RMSD, not the selection.

    # Use a concrete heuristic: for each motif residue j, score position i
    # by how close its mean distance to other positions is to gt mean distance.
    gt_mean_dist = jnp.mean(gt_distogram, axis=1)  # [M]
    pred_mean_dist = jnp.mean(pred_dist, axis=1)    # [N]

    topk_per_motif = []
    for j in range(M):
        score = -jnp.abs(pred_mean_dist - gt_mean_dist[j])
        _, idx = jax.lax.top_k(score, k)
        topk_per_motif.append(idx)

    # Cartesian product → [K^M, M]
    combos = jnp.array(list(product(*[range(k)] * M)))
    candidate_indices = jnp.stack(
        [topk_per_motif[j][combos[:, j]] for j in range(M)], axis=-1
    )

    # Evaluate RMSD for all candidates via vmap
    rmsds = jax.vmap(rmsd_at_indices, in_axes=(0, None, None))(
        candidate_indices, pred_ca, gt_ca
    )

    best_idx = jnp.argmin(rmsds)
    return rmsds[best_idx], candidate_indices[best_idx]


approach2_jit = jax.jit(partial(
    approach2_geometry, pred_ca=pred_ca, gt_ca=gt_ca, gt_distogram=gt_distogram, k=K,
))


# ===========================================================================
# Approach 3: Sequence-Guided Top-K
# ===========================================================================

def approach3_sequence_topk(pssm, gt_aa_indices, pred_ca, gt_ca, k=K):
    """
    1. For each motif residue, get top-K positions by PSSM probability of the
       correct amino acid.
    2. Enumerate K^M candidates, evaluate RMSD, pick best.
    3. Loss = best RMSD + sequence cross-entropy at selected positions.
    """
    topk_per_motif = []
    for j in range(M):
        aa_prob = pssm[:, gt_aa_indices[j]]  # [N]
        _, idx = jax.lax.top_k(aa_prob, k)
        topk_per_motif.append(idx)

    # Cartesian product → [K^M, M]
    combos = jnp.array(list(product(*[range(k)] * M)))
    candidate_indices = jnp.stack(
        [topk_per_motif[j][combos[:, j]] for j in range(M)], axis=-1
    )

    # vmap RMSD
    rmsds = jax.vmap(rmsd_at_indices, in_axes=(0, None, None))(
        candidate_indices, pred_ca, gt_ca
    )

    best_idx = jnp.argmin(rmsds)
    best_rmsd = rmsds[best_idx]
    best_positions = candidate_indices[best_idx]

    # sequence cross-entropy at selected positions
    gt_onehot = jax.nn.one_hot(gt_aa_indices, 20)  # [M, 20]
    sel_pssm = pssm[best_positions]                 # [M, 20]
    seq_loss = -jnp.mean(jnp.sum(gt_onehot * jnp.log(sel_pssm + 1e-8), axis=-1))

    return best_rmsd + 0.1 * seq_loss, best_positions


approach3_jit = jax.jit(partial(
    approach3_sequence_topk,
    pssm=pssm, gt_aa_indices=gt_aa_indices, pred_ca=pred_ca, gt_ca=gt_ca, k=K,
))


# ===========================================================================
# Approach 2b: Brute-force C(N, M) exhaustive search
# ===========================================================================

def approach2_bruteforce(pred_ca, gt_ca):
    """Enumerate ALL C(N, M) ordered index tuples and vmap RMSD."""
    from itertools import combinations
    all_combos = jnp.array(list(combinations(range(N), M)))  # [C(N,M), M]

    rmsds = jax.vmap(rmsd_at_indices, in_axes=(0, None, None))(
        all_combos, pred_ca, gt_ca
    )
    best_idx = jnp.argmin(rmsds)
    return rmsds[best_idx], all_combos[best_idx]


approach2_brute_jit = jax.jit(partial(
    approach2_bruteforce, pred_ca=pred_ca, gt_ca=gt_ca,
))


# ===========================================================================
# Run benchmarks
# ===========================================================================

if __name__ == "__main__":
    from itertools import combinations

    n_candidates_topk = K ** M
    n_candidates_brute = len(list(combinations(range(N), M)))

    print(f"Config: N={N}, M={M}, K={K}")
    print(f"  Top-K candidates per approach: K^M = {K}^{M} = {n_candidates_topk:,}")
    print(f"  Brute-force candidates: C({N},{M}) = {n_candidates_brute:,}")
    print(f"  Timing over {N_REPEATS} calls each\n")

    print("Approach 1a — Assignment Matrix (soft, no enumeration):")
    timeit(lambda: approach1_soft_jit(assignment_logits), label="soft weighted RMSD")

    print("\nApproach 1b — Assignment Matrix (top-K enumeration, weighted RMSD):")
    timeit(lambda: approach1_enum_jit(assignment_logits), label=f"K={K}, {n_candidates_topk:,} candidates")

    print("\nApproach 2a — Geometry-First (top-K distance matching):")
    timeit(lambda: approach2_jit(), label=f"K={K}, {n_candidates_topk:,} candidates")

    print("\nApproach 2b — Geometry-First (brute-force exhaustive):")
    timeit(lambda: approach2_brute_jit(), label=f"C({N},{M}) = {n_candidates_brute:,} candidates")

    print("\nApproach 3 — Sequence Top-K:")
    timeit(lambda: approach3_jit(), label=f"K={K}, {n_candidates_topk:,} candidates")

    # --- Results summary ---
    print(f"\n{'='*60}")
    print("Result verification (values from one call):")
    print(f"  1a soft RMSD:        {approach1_soft_jit(assignment_logits):.4f}")
    v1b = approach1_enum_jit(assignment_logits)
    print(f"  1b enum weighted:    {v1b:.4f}")
    v2a, idx2a = approach2_jit()
    print(f"  2a geom top-K:       {v2a:.4f}  indices={idx2a}")
    v2b, idx2b = approach2_brute_jit()
    print(f"  2b geom brute:       {v2b:.4f}  indices={idx2b}")
    v3, idx3 = approach3_jit()
    print(f"  3  seq top-K:        {v3:.4f}  indices={idx3}")
