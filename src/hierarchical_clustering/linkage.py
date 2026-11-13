"""Similarity-based linkage trees and hierarchical clustering objectives."""

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

METHODS = ("single", "complete", "average")


def validate_similarities(similarities):
    w = np.asarray(similarities, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or len(w) < 2:
        raise ValueError("Expected a square similarity matrix with at least two points")
    if not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("Similarities must be finite and nonnegative")
    if not np.array_equal(w, w.T) or np.any(w.diagonal() != 0):
        raise ValueError("Similarities must be symmetric with a zero diagonal")
    return w


def linkage_tree(similarities, method="average"):
    """Return (SciPy linkage matrix, root ClusterNode) for a similarity graph.

    C - w converts maximum/minimum/mean similarity to minimum/maximum/mean
    dissimilarity, respectively. Heights are dissimilarities, not LCA sizes.
    Leaves retain input row indices. Ties follow SciPy's tie-breaking rules.
    """

    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    w = validate_similarities(similarities)
    dissimilarities = w.max() - w
    np.fill_diagonal(dissimilarities, 0)
    merges = linkage(squareform(dissimilarities), method=method)
    return merges, to_tree(merges)


def evaluate_tree(similarities, merges):
    """Sum cost and revenue independently over each internal node's split.

    Every unordered pair occurs once, between the children of its LCA.
    Runtime is O(n^2); cluster memberships are discarded after each merge.
    """

    w = validate_similarities(similarities)
    n = len(w)
    merges = np.asarray(merges, dtype=float)
    root = to_tree(merges)  # Validate the linkage representation and counts.

    if root.count != n:
        raise ValueError("Tree and similarity matrix have different numbers of leaves")
    members = {i: [i] for i in range(n)}
    cost = revenue = 0.0

    for i, (left, right, _, _) in enumerate(merges):
        a, b = members.pop(int(left)), members.pop(int(right))
        size = len(a) + len(b)
        crossing_weight = float(w[np.ix_(a, b)].sum())
        cost += size * crossing_weight
        revenue += (n - size) * crossing_weight
        members[n + i] = a + b
    total_weight = float(w[np.triu_indices(n, 1)].sum())
    if not np.isclose(cost + revenue, n * total_weight):
        raise AssertionError("cost + revenue must equal n * total_weight")
    return {"cost": cost, "revenue": revenue, "total_weight": total_weight}
