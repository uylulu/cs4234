"""Root Max-Uncut bisection followed by average linkage or random splitting.

The outer constructions follow Ahmadian et al. and Alon--Azar--Vainstein,
but the basic-SDP bisection is heuristic and does not supply their guarantees.
"""

import numpy as np
from scipy.cluster.hierarchy import to_tree

from hierarchical_clustering.linkage import evaluate_tree, linkage_tree, validate_similarities
from hierarchical_clustering.max_uncut import _balanced_cut
from hierarchical_clustering.random_tree import random_binary_tree

CANDIDATE = "mub_average_candidate"
BEST_OF = "mub_average_best_of"
RANDOM = "mub_random"
METHODS = (CANDIDATE, BEST_OF, RANDOM)


def trees_from_bisection(similarities, left, right, *, seed=0):
    """Build both constructions from one balanced root partition.

    Also return the raw average-linkage candidate so the effect of choosing
    between it and global average linkage is visible. All trees preserve the
    original row IDs. Grafted subtree heights are leaf counts, not distances.
    """
    w = validate_similarities(similarities)
    n = len(w)
    a, b = np.asarray(left), np.asarray(right)
    if any(side.ndim != 1 or not np.issubdtype(side.dtype, np.integer) for side in (a, b)):
        raise ValueError("Partition sides must be one-dimensional integer arrays")
    if not len(a) or not len(b) or abs(len(a) - len(b)) > 1:
        raise ValueError("Root partition must be nonempty and balanced")
    if not np.array_equal(np.sort(np.concatenate((a, b))), np.arange(n)):
        raise ValueError("Partition must cover every vertex exactly once")
    a, b = np.sort(a), np.sort(b)
    child_seeds = np.random.SeedSequence(seed).spawn(2)

    def build(method):
        rows = []

        def graft(leaves, child_seed):
            if len(leaves) == 1:
                return int(leaves[0])
            if method == "average":
                local, _ = linkage_tree(w[np.ix_(leaves, leaves)], "average")
            else:
                local, _ = random_binary_tree(len(leaves), child_seed)
            mapping = dict(enumerate(map(int, leaves)))
            for i, (local_left, local_right, _, size) in enumerate(local):
                node_id = n + len(rows)
                rows.append([mapping[int(local_left)], mapping[int(local_right)], size, size])
                mapping[len(leaves) + i] = node_id
            return mapping[2 * len(leaves) - 2]

        left_root, right_root = graft(a, child_seeds[0]), graft(b, child_seeds[1])
        rows.append([left_root, right_root, n, n])
        merges = np.asarray(rows, dtype=float)
        to_tree(merges)  # Validate the remapped linkage representation.
        return merges

    candidate = build("average")
    random = build("random")
    global_average, _ = linkage_tree(w, "average")
    candidate_revenue = evaluate_tree(w, candidate)["revenue"]
    global_revenue = evaluate_tree(w, global_average)["revenue"]
    choose_global = global_revenue >= candidate_revenue
    # Make structural heights consistent across all three exported variants.
    global_average[:, 2] = global_average[:, 3]
    expected_random_revenue = sum(
        (n - (2 * len(side) + 2) / 3) * w[np.ix_(side, side)].sum() / 2 for side in (a, b)
    )
    return {
        CANDIDATE: candidate,
        BEST_OF: global_average if choose_global else candidate.copy(),
        RANDOM: random,
    }, {
        "left": a.tolist(),
        "right": b.tolist(),
        "selected": "global_average" if choose_global else "bisection_average",
        "global_average_revenue": global_revenue,
        "candidate_revenue": candidate_revenue,
        "conditional_expected_random_revenue": float(expected_random_revenue),
    }


def max_uncut_hybrid_trees(
    similarities,
    seed=0,
    *,
    rounds=32,
    solver_eps=1e-5,
    solver_max_iters=20_000,
):
    """Solve one root SDP, then return the raw AL, best-of AL, and random trees."""
    w = validate_similarities(similarities)
    (a, b), diagnostics = _balanced_cut(
        w, np.random.default_rng(seed), rounds, solver_eps, solver_max_iters
    )
    trees, info = trees_from_bisection(w, a, b, seed=seed)
    return trees, {**diagnostics, **info}
