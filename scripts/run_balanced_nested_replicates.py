"""Benchmark seven methods on ten independent balanced-nested graphs."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import squareform

from hierarchical_clustering.linkage import METHODS, evaluate_tree, linkage_tree
from hierarchical_clustering.max_uncut import experiment_settings, round_basic_sdp, solve_basic_sdp
from hierarchical_clustering.max_uncut_hybrids import CANDIDATE, RANDOM, trees_from_bisection
from hierarchical_clustering.random_tree import random_binary_tree
from hierarchical_clustering.spectral import spectral_tree
from hierarchical_clustering.synthetic import generate_synthetic

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "results/balanced_nested_replicates"
    data = ROOT / "data/processed/balanced_nested_replicates"
    output.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    rows, diagnostics, manifest = [], {}, []
    start = perf_counter()
    for generation_seed in range(4234, 4244):
        dataset = f"balanced_nested_{generation_seed}"
        arrays, metadata = generate_synthetic("balanced_nested", n=128, seed=generation_seed)
        w = arrays["similarities"]
        if generation_seed == 4234:
            original = np.load(data / f"balanced_nested_{generation_seed}.npz")
            np.testing.assert_array_equal(w, original["similarities"])
        path = data / f"{dataset}.npz"
        np.savez_compressed(path, **arrays)
        metadata["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata["path"] = str(path.relative_to(ROOT))
        manifest.append(metadata)
        trees, methods, seeds = [], [], []
        weights = squareform(w)

        def record(method, seed, tree):
            score = evaluate_tree(w, tree)
            structural = tree.copy()
            structural[:, 2] = structural[:, 3]
            independent_cost = float(weights @ cophenet(structural))
            np.testing.assert_allclose(score["cost"], independent_cost, rtol=1e-12)
            np.testing.assert_allclose(
                score["revenue"], len(w) * score["total_weight"] - independent_cost,
                rtol=1e-12,
            )
            bound = (len(w) - 2) * score["total_weight"]
            rows.append({
                "dataset": dataset, "generation_seed": generation_seed, "method": method,
                "seed": seed, "n": len(w), **score, "revenue_upper_bound": bound,
                "revenue_over_upper_bound": score["revenue"] / bound,
            })
            trees.append(tree)
            methods.append(method)
            seeds.append(seed)

        for method in METHODS:
            record(method, -1, linkage_tree(w, method)[0])
        record("spectral", -1, spectral_tree(w)[0])
        for seed in range(4234, 4334):
            record("random", seed, random_binary_tree(len(w), seed)[0])
        solution = solve_basic_sdp(w)
        diagnostics[dataset] = {"solve": solution.diagnostics, "runs": {}}
        for seed in range(4234, 4254):
            (a, b), info = round_basic_sdp(w, solution, seed=seed, rounds=32)
            variants, selection = trees_from_bisection(w, a, b, seed=seed)
            diagnostics[dataset]["runs"][str(seed)] = {**info, **selection}
            for method in (CANDIDATE, RANDOM):
                record(method, seed, variants[method])
        np.savez_compressed(output / f"{dataset}_trees.npz", trees=np.asarray(trees),
                            methods=np.asarray(methods), seeds=np.asarray(seeds))
        # Checkpoint each completed graph as well as the final consolidated results.
        pd.DataFrame(rows).to_csv(output / "scores.csv", index=False)
        (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        print(f"{dataset}: {len(trees)} trees scored and independently checked; "
              f"elapsed {perf_counter() - start:.1f}s", flush=True)

    scores = pd.DataFrame(rows)
    assert len(scores) == 1440 and scores.dataset.nunique() == 10
    metrics = ["cost", "revenue", "revenue_over_upper_bound"]
    per_graph = scores.groupby(["dataset", "generation_seed", "method"])[metrics].mean()
    per_graph.to_csv(output / "per_graph.csv")
    summary = per_graph.groupby("method")[metrics].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.to_csv(output / "summary.csv")
    (output / "metadata.json").write_text(json.dumps({
        "generation_seeds": list(range(4234, 4244)), "graphs": manifest,
        "random_tree_seeds": list(range(4234, 4334)),
        "hybrid_seeds": list(range(4234, 4254)),
        "sdp_settings": experiment_settings(32, 1e-5, 20_000),
        "root_sdp_reuse": "One solve per graph, shared by both hybrid methods",
        "aggregation": "Mean across runs within graph, then mean and sample SD across 10 graphs",
        "validation": "All 1440 trees independently scored using SciPy pairwise LCA sizes",
        "original_instance": "Seed 4234 exactly matches the original saved input",
        "elapsed_seconds": perf_counter() - start,
    }, indent=2) + "\n")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
