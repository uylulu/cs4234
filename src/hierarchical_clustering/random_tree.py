"""Random hierarchies from independent fair-coin splits."""

import numpy as np
from scipy.cluster.hierarchy import to_tree


def random_binary_tree(n, seed=0):
    """Return a SciPy linkage matrix and root for n labelled leaves.

    Assign each point independently to either side with probability 1/2,
    retrying empty splits. Recurse until singleton leaves. This is not uniform
    sampling over all binary tree topologies, nor forced balanced splitting.
    Heights are structural (subtree size), not distances or similarities.
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 2:
        raise ValueError("n must be an integer of at least 2")
    rng = np.random.default_rng(seed)
    rows = []

    def split(leaves):
        if len(leaves) == 1:
            return int(leaves[0])
        while True:
            mask = rng.integers(0, 2, size=len(leaves)).astype(bool)
            if mask.any() and not mask.all():
                break
        left = split(leaves[mask])
        right = split(leaves[~mask])
        node_id = n + len(rows)
        rows.append([left, right, len(leaves), len(leaves)])
        return node_id

    split(np.arange(n))
    merges = np.asarray(rows, dtype=float)
    return merges, to_tree(merges)
