"""Compile seven requested methods; complete cross-pair random subtrees.

Reuse archived basic-SDP root partitions, preserving the original rounding
experiment. Existing result archives are read only.
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import squareform

from hierarchical_clustering.linkage import evaluate_tree
from hierarchical_clustering.max_uncut import round_basic_sdp, solve_basic_sdp
from hierarchical_clustering.max_uncut_hybrids import CANDIDATE, RANDOM, trees_from_bisection

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "single": "1. Single linkage",
    "complete": "2. Complete linkage",
    "average": "3. Average linkage",
    "random": "4. Random binary tree",
    "spectral": "5. Recursive spectral cut",
    CANDIDATE: "7. Max-Uncut → average linkage",
    RANDOM: "8. Max-Uncut → random splits",
}
DATASETS = {
    "iris": "Iris (150)",
    "wine": "Wine (178)",
    "balanced_nested": "Balanced nested (128)",
    "cross_pairs": "Cross-pair stress (32)",
}


def main():
    output = ROOT / "results/selected_algorithms"
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}

    def source(name):
        path = ROOT / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    hybrid_frames = []
    for dataset in ("iris", "wine"):
        w = np.load(source(f"data/processed/{dataset}.npz"))["similarities"]
        solution = solve_basic_sdp(np.pad(w, ((0, len(w) % 2), (0, len(w) % 2))))
        rows, trees, methods, seeds = [], [], [], []
        for seed in range(4234, 4254):
            (left, right), _ = round_basic_sdp(w, solution, seed=seed, rounds=32)
            variants, _ = trees_from_bisection(w, left, right, seed=seed)
            for method in (CANDIDATE, RANDOM):
                score = evaluate_tree(w, variants[method])
                rows.append({"dataset": dataset, "method": method, "seed": seed,
                             "n": len(w), **score, "revenue_upper_bound": (len(w)-2)*score["total_weight"],
                             "revenue_over_upper_bound": score["revenue"] / ((len(w)-2)*score["total_weight"])})
                trees.append(variants[method])
                methods.append(method)
                seeds.append(seed)
        np.savez_compressed(output / f"{dataset}_root_trees.npz", similarities=w, trees=trees, methods=methods, seeds=seeds)
        hybrid_frames.append(pd.DataFrame(rows))

    cross_rows = []
    for generation_seed in range(5000, 5010):
        dataset = f"cross_pairs_{generation_seed}"
        path = source(f"data/processed/exploratory_cross_pairs/{dataset}.npz")
        w = np.load(path)["similarities"]
        solution = solve_basic_sdp(w)
        trees, methods, seeds = [], [], []
        for seed in range(20):
            (left, right), _ = round_basic_sdp(w, solution, seed=seed, rounds=32)
            variants, _ = trees_from_bisection(w, left, right, seed=seed)
            for method in (CANDIDATE, RANDOM):
                tree = variants[method]
                score = evaluate_tree(w, tree)
                structural = tree.copy()
                structural[:, 2] = structural[:, 3]
                pairwise_cost = float(squareform(w) @ cophenet(structural))
                np.testing.assert_allclose(score["cost"], pairwise_cost, rtol=1e-12)
                np.testing.assert_allclose(
                    score["revenue"], len(w) * score["total_weight"] - pairwise_cost,
                    rtol=1e-12,
                )
                bound = (len(w) - 2) * score["total_weight"]
                cross_rows.append({
                    "dataset": dataset, "generation_seed": int(dataset.rsplit("_", 1)[1]),
                    "method": method, "seed": seed, "n": len(w), **score,
                    "revenue_upper_bound": bound,
                    "revenue_over_upper_bound": score["revenue"] / bound,
                })
                trees.append(tree)
                methods.append(method)
                seeds.append(seed)
        np.savez_compressed(
            output / f"{dataset}_root_trees.npz", similarities=w,
            trees=np.asarray(trees), methods=np.asarray(methods), seeds=np.asarray(seeds),
        )
    cross = pd.DataFrame(cross_rows)
    cross.to_csv(output / "cross_pair_root_scores.csv", index=False)

    frames = []
    for name in ("linkage", "random", "spectral"):
        frames.append(pd.read_csv(source(f"results/{name}/scores.csv")))
    nested = pd.read_csv(source("results/balanced_nested_replicates/scores.csv"))
    assert nested.dataset.nunique() == 10 and len(nested) == 1440
    frames.append(nested)
    frames.extend(hybrid_frames)
    frames.extend([pd.read_csv(source("results/exploratory_cross_pairs/scores.csv")), cross])
    scores = pd.concat(frames, ignore_index=True)
    scores = scores.loc[scores.method.isin(METHODS)].copy()
    scores["family"] = scores.dataset.where(~scores.dataset.str.startswith("cross_pairs_"),
                                            "cross_pairs")
    scores.loc[scores.dataset.str.startswith("balanced_nested_"), "family"] = "balanced_nested"
    scores["revenue_upper_bound"] = (scores.n - 2) * scores.total_weight
    np.testing.assert_allclose(scores.cost + scores.revenue, scores.n * scores.total_weight)
    np.testing.assert_allclose(scores.revenue_over_upper_bound,
                               scores.revenue / scores.revenue_upper_bound)
    columns = ["family", "dataset", "method", "seed", "n", "cost", "revenue",
               "total_weight", "revenue_upper_bound", "revenue_over_upper_bound"]
    scores[columns].to_csv(output / "scores.csv", index=False)
    metrics = ["cost", "revenue", "revenue_over_upper_bound"]
    # Equal weight per graph, after averaging algorithm repetitions within each graph.
    per_graph = scores.groupby(["family", "dataset", "method"], sort=False)[metrics].mean()
    per_graph.to_csv(output / "per_graph.csv")
    summary = per_graph.groupby(["family", "method"], sort=False)[metrics].mean()
    summary["graphs"] = per_graph.groupby(["family", "method"]).size()
    summary["tree_runs"] = scores.groupby(["family", "method"]).size()
    summary["ratio_between_graph_sd"] = per_graph.groupby(["family", "method"])[
        "revenue_over_upper_bound"
    ].std()
    assert len(summary) == 28
    for family in ("balanced_nested", "cross_pairs"):
        assert (summary.loc[family, "graphs"] == 10).all()
    summary.to_csv(output / "summary.csv")

    def table(metric, digits):
        lines = ["| Algorithm | " + " | ".join(DATASETS.values()) + " |",
                 "|---|" + "---:|" * len(DATASETS)]
        for method, label in METHODS.items():
            values = [summary.loc[(dataset, method), metric] for dataset in DATASETS]
            lines.append(f"| {label} | " + " | ".join(
                f"{value:,.{digits}f}" for value in values) + " |")
        return "\n".join(lines)

    variation = ["| Algorithm | Balanced nested: mean ± SD | Cross-pair: mean ± SD |",
                 "|---|---:|---:|"]
    for method, label in METHODS.items():
        cells = []
        for family in ("balanced_nested", "cross_pairs"):
            row = summary.loc[(family, method)]
            cells.append(f"{row[metrics[2]]:.6f} ± {row.ratio_between_graph_sd:.6f}")
        variation.append(f"| {label} | " + " | ".join(cells) + " |")

    fig, axes = plt.subplots(1, 4, figsize=(17, 5.8), sharey=True)
    colors = ["#7e9ac2", "#a2b8d5", "#23649c", "#b5b5b5", "#e49b32", "#279483", "#84bdae"]
    for ax, (dataset, label) in zip(axes, DATASETS.items(), strict=True):
        values = [summary.loc[(dataset, method), metrics[2]] for method in METHODS]
        errors = [summary.loc[(dataset, method), "ratio_between_graph_sd"] for method in METHODS]
        ax.barh(list(METHODS.values()), values, color=colors,
                xerr=np.nan_to_num(errors), capsize=2, error_kw={"elinewidth": 1})
        for y, value in enumerate(values):
            ax.text(value + .008, y, f"{value:.4f}", va="center", fontsize=9)
        ax.set(title=label, xlim=(0, .73), xlabel="Revenue / upper bound (higher is better)")
        ax.axvline(1 / 3, color="gray", linestyle=":", linewidth=1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle("Seven algorithms on the four selected datasets", fontsize=15)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output / f"revenue_comparison.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = """# Selected algorithm experiment results

Seven requested methods on Iris, Wine, balanced nested, and cross-pair stress.
Methods 7 and 8 use one basic-SDP Max-Uncut root bisection (CVXPY/SCS,
best of 32 rounding candidates). Method 7 is the raw root-bisection →
average-linkage construction. It does not choose between this tree and global
average linkage. The basic SDP and spectral sweep are practical heuristics;
these experiments do not establish the certified algorithms' approximation factors.

## Revenue divided by its upper bound

**Table 1** reports the mean normalized revenue $R/U$ for each algorithm and
dataset; higher values indicate better revenue. For each graph, $W=\\sum_{i<j}w_{ij}$
and $U=(n-2)W$. Each entry averages algorithm runs within a graph and, for the
synthetic families, then averages equally across the ten graphs. It reports
$R/U$, not $R/\\mathrm{OPT}$; since $U$ upper-bounds OPT, this is a conservative
lower bound on the true approximation ratio.

""" + table(metrics[2], 6) + """

![Revenue comparison](results/selected_algorithms/revenue_comparison.png)

## Dasgupta cost

**Table 2** reports mean Dasgupta cost $C(T)$ for the same algorithms and datasets;
lower values are better. Synthetic entries use the same two-stage averaging as
Table 1. Costs are comparable between algorithms on one dataset, while their
magnitudes also depend on the dataset size and total similarity weight.

""" + table("cost", 2) + """

## Revenue

**Table 3** reports mean Moseley–Wang revenue $R(T)$ in the original weight scale;
higher values are better. It uses the same averaging as Tables 1 and 2. The identity
$C(T)+R(T)=nW$ holds for every tree, so cost and revenue give equivalent rankings
within a fixed dataset; this table shows the unnormalized revenue values.

""" + table("revenue", 2) + """

## Variation across synthetic graphs

Both families have ten independently generated graphs. These are means ± sample
standard deviations of the per-graph mean revenue ratios, not confidence intervals
or the standard deviation of all pooled algorithm runs. Plot error bars show these SDs.

""" + "\n".join(variation) + """

## Sampling and dataset details

- Iris: 150 observations; Wine: 178 observations. Numerical features were
  standardized separately within each dataset, then converted to dense Gaussian
  similarities using the median strictly positive pairwise distance as bandwidth.
  Labels were excluded and the diagonal was set to zero.
- Balanced nested: ten 128-node graphs, generation seeds 4234–4243. Eight leaf groups
  form a balanced three-level hierarchy, with higher weights inside deeper groups;
  independent edge noise has standard deviation 0.02.
- Cross-pair stress: ten 32-node graphs, generation seeds 5000–5009. Each graph has
  two groups of 16 nodes, within-group weights 0.6, a perfect cross-group matching
  of weight 1, and other cross-group weights 0.02, before edge noise of standard
  deviation 0.01 and clipping. This is a deliberately constructed stress test.
- Iris, Wine, and each balanced-nested graph: deterministic methods have one run; random
  binary trees average 100 runs (seeds 4234–4333); methods 7 and 8 average 20 runs
  (seeds 4234–4253).
- Cross-pair stress: deterministic methods run once per graph; random trees and
  methods 7 and 8 average 20 algorithm seeds (0–19) per graph. We first average
  within each graph, then equally across the ten graphs. The ratio column is the
  mean of individual graph ratios. Random results are sampled means, not exact
  expectations. Standalone random splitting uses independent fair coin flips,
  resampling empty splits; the root in method 8 is balanced.

## Provenance and validation

Iris and Wine baseline and root-bisection results were reused. Balanced nested
was rerun on ten graphs using the same seven algorithms, with all 1,440 trees
independently scored using pairwise LCA sizes. Seed 4234 reproduces the original
saved balanced-nested input exactly. Cross-pair baseline results were reused;
missing cross-pair
random subtrees were generated from the saved root partitions in
`results/best_of_validation/diagnostics.json`, with the original algorithm seeds.
All 200 regenerated root-average trees exactly matched their saved tree arrays.
All 400 cross-pair hybrid trees were independently rescored from pairwise LCA
sizes using SciPy. Every compiled score satisfied $C+R=nW$ and the ratio formula.

Generate the balanced-nested results with
`.venv/bin/python scripts/run_balanced_nested_replicates.py`, then rebuild this
comparison with `.venv/bin/python scripts/selected_algorithm_results.py`.
Full scores, per-graph means, summary, tree archives, and input SHA-256 hashes
are in `results/selected_algorithms/`.
"""
    (ROOT / "SELECTED_ALGORITHM_RESULTS.md").write_text(report)
    (output / "metadata.json").write_text(json.dumps({
        "source_sha256": hashes, "compiled_tree_scores": len(scores),
        "cross_pair_hybrid_trees_checked": len(cross),
        "cross_pair_candidate_trees_matched": len(cross) // 2,
        "aggregation": "Mean within each graph, then equal-weight mean over graphs",
        "max_uncut": "Basic SDP; archived root partitions; 32 rounding candidates",
    }, indent=2) + "\n")
    print("TABLE 1 — Mean revenue divided by U=(n−2)W (higher is better).")
    print("Synthetic entries average over ten graphs; this is a lower bound on R/OPT, not R/OPT itself.\n")
    print(table(metrics[2], 6))
    print("\nTABLE 2 — Mean Dasgupta cost C(T) (lower is better).")
    print("Raw costs depend on graph size and total edge weight.\n")
    print(table("cost", 2))
    print("\nTABLE 3 — Mean Moseley–Wang revenue R(T) in original weight units (higher is better).")
    print("For each dataset, C(T)+R(T)=nW, so cost and revenue give equivalent rankings.\n")
    print(table("revenue", 2))
    print(f"\nValidated {len(cross)} hybrid trees; compiled {len(scores)} score rows.")


if __name__ == "__main__":
    main()
