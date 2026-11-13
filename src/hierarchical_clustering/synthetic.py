"""Configurable synthetic similarity graphs; no feature scaling or Gaussian kernel.

Run `python -m hierarchical_clustering.synthetic` to save five examples.
"""

import json
from pathlib import Path

import numpy as np

from hierarchical_clustering.linkage import validate_similarities

DEFAULTS = {
    "balanced_nested": {"depth": 3, "low": 0.1, "high": 0.9},
    "unbalanced_nested": {"depth": 3, "split_fraction": 0.75, "low": 0.1, "high": 0.9},
    "chains_bridges": {
        "groups": 4,
        "bridge_length": 6,
        "within": 0.9,
        "background": 0.03,
        "bridge_weight": 0.8,
    },
    "overlapping": {"groups": 4, "overlap_fraction": 0.25, "background": 0.05},
    "core_periphery": {
        "groups": 4,
        "core_fraction": 0.2,
        "core_within": 0.8,
        "periphery_within": 0.9,
        "core_to_periphery": 0.6,
        "between_peripheries": 0.05,
    },
}


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _unit(value, name):
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and between 0 and 1")


def _nested(n, depth, low, high, split_fraction=0.5):
    _integer(depth, "depth")
    if n < 2**depth:
        raise ValueError("n must be at least 2**depth")
    if not 0.5 <= split_fraction < 1:
        raise ValueError("split_fraction must be in [0.5, 1)")
    if low > high:
        raise ValueError("low must not exceed high")
    hierarchy = np.zeros((n, depth), dtype=int)

    def split(indices, level, prefix):
        if level == depth:
            return
        # Reserve enough points for all remaining nonempty descendant groups.
        minimum = 2 ** (depth - level - 1)
        size = int(np.clip(round(len(indices) * split_fraction), minimum, len(indices) - minimum))
        for bit, child in enumerate((indices[:size], indices[size:])):
            group = 2 * prefix + bit
            hierarchy[child, level] = group
            split(child, level + 1, group)

    split(np.arange(n), 0, 0)
    shared_levels = (hierarchy[:, None, :] == hierarchy[None, :, :]).sum(axis=2)
    w = low + (high - low) * shared_levels / depth
    return {"similarities": w, "labels": hierarchy[:, -1], "hierarchy": hierarchy}


def _chains(n, groups, bridge_length, within, background, bridge_weight):
    _integer(groups, "groups", 2)
    _integer(bridge_length, "bridge_length")
    bridge_count = (groups - 1) * bridge_length
    if n - bridge_count < 2 * groups:
        raise ValueError("Need at least two cluster points per group in addition to bridges")
    clusters = np.array_split(np.arange(n - bridge_count), groups)
    w = np.full((n, n), background)
    labels = np.full(n, -1, dtype=int)
    bridge_ids = np.full(n, -1, dtype=int)
    positions = np.full(n, -1, dtype=int)
    for group, indices in enumerate(clusters):
        labels[indices] = group
        w[np.ix_(indices, indices)] = within
    for group in range(groups - 1):
        chain = np.arange(
            n - bridge_count + group * bridge_length, n - bridge_count + (group + 1) * bridge_length
        )
        bridge_ids[chain], positions[chain] = group, np.arange(bridge_length)
        w[chain[:-1], chain[1:]] = w[chain[1:], chain[:-1]] = bridge_weight
        w[chain[0], clusters[group]] = w[clusters[group], chain[0]] = bridge_weight
        w[chain[-1], clusters[group + 1]] = w[clusters[group + 1], chain[-1]] = bridge_weight
    return {
        "similarities": w,
        "labels": labels,
        "bridge_ids": bridge_ids,
        "bridge_positions": positions,
        "roles": np.where(labels < 0, "bridge", "cluster"),
    }


def _overlap(n, rng, groups, overlap_fraction, background):
    _integer(groups, "groups", 2)
    mixed_count = round(n * overlap_fraction)
    pure_count = n - mixed_count
    if pure_count < groups:
        raise ValueError("Need at least one pure point per group")
    membership = np.zeros((n, groups))
    for group, indices in enumerate(np.array_split(np.arange(pure_count), groups)):
        membership[indices, group] = 1
    for i in range(pure_count, n):
        membership[i, rng.choice(groups, size=2, replace=False)] = 0.5
    unit_membership = membership / np.linalg.norm(membership, axis=1, keepdims=True)
    w = background + (1 - background) * (unit_membership @ unit_membership.T)
    labels = np.argmax(membership, axis=1)
    labels[pure_count:] = -1  # No artificial single-class truth for mixed members.
    return {
        "similarities": w,
        "labels": labels,
        "memberships": membership,
        "roles": np.where(labels < 0, "mixed", "pure"),
    }


def _core(
    n, groups, core_fraction, core_within, periphery_within, core_to_periphery, between_peripheries
):
    _integer(groups, "groups", 2)
    core_count = round(n * core_fraction)
    if core_count < 1 or n - core_count < groups:
        raise ValueError("Need a nonempty core and each peripheral group")
    labels = np.zeros(n, dtype=int)
    w = np.full((n, n), between_peripheries)
    w[:core_count, :core_count] = core_within
    w[:core_count, core_count:] = w[core_count:, :core_count] = core_to_periphery
    for group, indices in enumerate(np.array_split(np.arange(core_count, n), groups), start=1):
        labels[indices] = group
        w[np.ix_(indices, indices)] = periphery_within
    return {
        "similarities": w,
        "labels": labels,
        "roles": np.where(labels == 0, "core", "periphery"),
    }


def generate_synthetic(family, n=128, seed=4234, noise=0.02, shuffle=True, **parameters):
    """Return (arrays, metadata); all node-aligned arrays share shuffled row order.

    `similarities` is algorithm input; `planted_similarities` is the noise-free
    reference. Labels/memberships describe the planted structure, not proven OPT.
    Noise is independent N(0, noise**2) on unordered pairs, clipped to [0,1].
    `display_order` reverses shuffling for structured heatmap inspection only.
    """
    if family not in DEFAULTS:
        raise ValueError(f"Unknown family {family!r}; choose from {tuple(DEFAULTS)}")
    _integer(n, "n", 2)
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    if parameters.keys() - DEFAULTS[family].keys():
        raise ValueError(f"Unknown parameters: {parameters.keys() - DEFAULTS[family].keys()}")
    config = DEFAULTS[family] | parameters
    for key, value in config.items():
        if key not in ("depth", "groups", "bridge_length"):
            _unit(value, key)
    # Separate streams keep the same perturbations/order when structure parameters change.
    structure_seed, noise_seed, order_seed = np.random.SeedSequence(seed).spawn(3)
    if family in ("balanced_nested", "unbalanced_nested"):
        arrays = _nested(n, **config)
    elif family == "chains_bridges":
        arrays = _chains(n, **config)
    elif family == "overlapping":
        arrays = _overlap(n, np.random.default_rng(structure_seed), **config)
    else:
        arrays = _core(n, **config)
    reference = np.clip(arrays.pop("similarities"), 0, 1)
    np.fill_diagonal(reference, 0)
    i, j = np.triu_indices(n, 1)
    perturbed = np.zeros_like(reference)
    values = np.clip(
        reference[i, j] + np.random.default_rng(noise_seed).normal(0, noise, len(i)), 0, 1
    )
    perturbed[i, j] = perturbed[j, i] = values
    permutation = np.random.default_rng(order_seed).permutation(n) if shuffle else np.arange(n)
    arrays = {key: value[permutation] for key, value in arrays.items()}
    arrays.update(
        similarities=perturbed[np.ix_(permutation, permutation)],
        planted_similarities=reference[np.ix_(permutation, permutation)],
        original_indices=permutation,
        display_order=np.argsort(permutation),
    )
    validate_similarities(arrays["similarities"])
    metadata = {
        "family": family,
        "n": n,
        "seed": seed,
        "noise_std": noise,
        "shuffle": shuffle,
        "parameters": config,
        "numpy_version": np.__version__,
        "representation": "direct similarity graph; no Gaussian bandwidth or scaling",
        "noise": "independent Gaussian perturbations on unordered pairs; clip to [0,1]",
        "labels": "planted membership only; -1 indicates mixed/bridge points",
        "optimum": "not computed; planted structure is not a certificate of OPT",
    }
    return arrays, metadata


def save_synthetic_families(root, n=128, seed=4234, noise=0.02):
    """Save one example per family under data/processed/synthetic/."""
    output = Path(root) / "data" / "processed" / "synthetic"
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for family in DEFAULTS:
        arrays, metadata = generate_synthetic(family, n=n, seed=seed, noise=noise)
        np.savez_compressed(output / f"{family}.npz", **arrays)
        (output / f"{family}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        manifest.append(metadata)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    for info in save_synthetic_families(Path(__file__).resolve().parents[2]):
        print(f"{info['family']}: n={info['n']}, seed={info['seed']}, noise={info['noise_std']}")
