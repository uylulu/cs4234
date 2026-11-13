"""Recursive Fiedler-vector sweep heuristic for similarity graphs."""

import numpy as np
from scipy.cluster.hierarchy import to_tree
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components, laplacian

from hierarchical_clustering.linkage import validate_similarities


def _sweep_cut(w, order):
    """Minimize uniform sparsity over all nonempty prefixes in O(s^2)."""
    ordered = w[np.ix_(order, order)]
    # Moving vertex k into A adds its edges to B and removes its edges to A.
    increments = ordered.sum(axis=1) - 2 * np.tril(ordered, k=-1).sum(axis=1)
    crossing = np.maximum(np.cumsum(increments)[:-1], 0.0)
    sizes = np.arange(1, len(w))
    sparsity = crossing / (sizes * (len(w) - sizes))
    cut = int(np.argmin(sparsity)) + 1
    return order[:cut], order[cut:]


def _partition(w):
    n = len(w)
    components, labels = connected_components(w, directed=False)
    if components > 1:
        # Separate whole components: crossing weight is exactly zero.
        order = np.argsort(labels, kind="stable")
        boundaries = np.cumsum(np.bincount(labels))[:-1]
        cut = boundaries[np.argmin(np.abs(boundaries - n / 2))]
        return order[:cut], order[cut:]
    _, vectors = eigh(laplacian(w, normed=False), subset_by_index=[1, 1])
    fiedler = vectors[:, 0]
    # Fix eigenvector sign; stable sorting resolves equal coordinates by row order.
    if fiedler[np.argmax(np.abs(fiedler))] < 0:
        fiedler = -fiedler
    return _sweep_cut(w, np.argsort(fiedler, kind="stable"))


def spectral_tree(similarities):
    """Return (linkage-format matrix, root) using recursive spectral cuts.

    At each node form the induced unnormalized Laplacian D-W, sort its second
    eigenvector, and minimize w(A,B)/(|A||B|) over prefix cuts. This is a
    heuristic, not a certified sparsest-cut approximation. No forced balance.
    Heights encode subtree sizes, not distances. Degenerate eigenspaces can
    yield different trees across LAPACK/library versions.
    """
    w = validate_similarities(similarities)
    n = len(w)
    rows = []

    def split(leaves):
        if len(leaves) == 1:
            return int(leaves[0])
        a, b = _partition(w[np.ix_(leaves, leaves)])
        left, right = split(leaves[a]), split(leaves[b])
        node_id = n + len(rows)
        rows.append([left, right, len(leaves), len(leaves)])
        return node_id

    split(np.arange(n))
    merges = np.asarray(rows, dtype=float)
    return merges, to_tree(merges)
