"""Exploratory dense-block graphs with stronger, misleading cross-group pairs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hierarchical_clustering.linkage import METHODS, evaluate_tree, linkage_tree
from hierarchical_clustering.max_uncut import experiment_settings, max_uncut_tree
from hierarchical_clustering.random_tree import random_binary_tree
from hierarchical_clustering.spectral import spectral_tree


def cross_pair_graph(n=32, *, seed=5000, noise=0.01):
    """Two equal dense blocks with a stronger perfect matching across blocks.

    This is an explicitly designed exploratory stress test, not a random sample
    of typical inputs. Labels and construction order are saved only for analysis.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 4 or n % 2:
        raise ValueError("n must be an even integer >=4")
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    half = n // 2
    labels = np.repeat([0, 1], half)
    reference = np.where(labels[:, None] == labels[None, :], 0.6, 0.02)
    i = np.arange(half)
    reference[i, i + half] = reference[i + half, i] = 1.0
    np.fill_diagonal(reference, 0)
    noise_seed, order_seed = np.random.SeedSequence(seed).spawn(2)
    upper = np.triu_indices(n, 1)
    w = np.zeros_like(reference)
    values = np.clip(
        reference[upper] + np.random.default_rng(noise_seed).normal(0, noise, len(upper[0])),
        0,
        1,
    )
    w[upper] = values
    w[(upper[1], upper[0])] = values
    permutation = np.random.default_rng(order_seed).permutation(n)
    return {
        "similarities": w[np.ix_(permutation, permutation)],
        "planted_similarities": reference[np.ix_(permutation, permutation)],
        "labels": labels[permutation],
        "original_indices": permutation,
        "display_order": np.argsort(permutation),
    }


def run_cross_pair_experiments(root, *, generation_seeds=range(5000, 5010), seeds=range(20)):
    root = Path(root)
    data = root / "data/processed/exploratory_cross_pairs"
    output = root / "results/exploratory_cross_pairs"
    data.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    generation_seeds, seeds = list(generation_seeds), list(seeds)
    rows, diagnostics, hashes = [], {}, {}
    for generation_seed in generation_seeds:
        arrays = cross_pair_graph(seed=generation_seed)
        w = arrays["similarities"]
        name = f"cross_pairs_{generation_seed}"
        source = data / f"{name}.npz"
        np.savez_compressed(source, **arrays)
        hashes[str(source.relative_to(root))] = hashlib.sha256(source.read_bytes()).hexdigest()
        trees, methods, tree_seeds = [], [], []

        def record(method, seed, tree):
            scores = evaluate_tree(w, tree)
            rows.append(
                {
                    "dataset": name,
                    "generation_seed": generation_seed,
                    "method": method,
                    "seed": seed,
                    "n": len(w),
                    **scores,
                    "revenue_over_upper_bound": scores["revenue"]
                    / ((len(w) - 2) * scores["total_weight"]),
                }
            )
            trees.append(tree)
            methods.append(method)
            tree_seeds.append(seed)

        for method in METHODS:
            record(method, -1, linkage_tree(w, method)[0])
        record("spectral", -1, spectral_tree(w)[0])
        diagnostics[name] = {}
        for seed in seeds:
            record("random", seed, random_binary_tree(len(w), seed)[0])
            tree, _, info = max_uncut_tree(w, seed, return_diagnostics=True)
            record("max_uncut_sdp", seed, tree)
            diagnostics[name][str(seed)] = info
        np.savez_compressed(
            output / f"{name}_trees.npz",
            trees=np.asarray(trees),
            methods=np.asarray(methods),
            seeds=np.asarray(tree_seeds),
        )
        print(f"Finished exploratory graph {generation_seed}", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(output / "scores.csv", index=False)
    per_graph = (
        scores.groupby(["generation_seed", "method"], sort=False)
        .agg(
            runs=("revenue", "size"),
            cost=("cost", "mean"),
            revenue=("revenue", "mean"),
            revenue_over_upper_bound=("revenue_over_upper_bound", "mean"),
            within_graph_ratio_sd=("revenue_over_upper_bound", "std"),
        )
        .reset_index()
    )
    per_graph.to_csv(output / "per_graph.csv", index=False)
    per_graph.groupby("method").agg(
        graphs=("revenue", "size"),
        cost_mean=("cost", "mean"),
        revenue_mean=("revenue", "mean"),
        ratio_mean=("revenue_over_upper_bound", "mean"),
        ratio_between_graph_sd=("revenue_over_upper_bound", "std"),
        ratio_min=("revenue_over_upper_bound", "min"),
        ratio_max=("revenue_over_upper_bound", "max"),
    ).to_csv(output / "summary.csv")
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "design": "exploratory family designed after the original five-family benchmark",
                "n": 32,
                "block_sizes": [16, 16],
                "within": 0.6,
                "cross_matching": 1.0,
                "other_cross": 0.02,
                "noise_std": 0.01,
                "generation_seeds": generation_seeds,
                "algorithm_seeds": seeds,
                "noise": "independent unordered-pair Gaussian noise; clip to [0,1]; shuffle rows",
                "input_sha256": hashes,
                "max_uncut": experiment_settings(32, 1e-5, 20_000),
                "uncertainty": "aggregate SD is across graph means; algorithm seeds are not independent graphs",
                "limitations": "targeted stress test at one parameter setting; no claim of typical performance or known OPT",
            },
            indent=2,
        )
        + "\n"
    )
    return scores


if __name__ == "__main__":
    run_cross_pair_experiments(Path(__file__).resolve().parents[2])
