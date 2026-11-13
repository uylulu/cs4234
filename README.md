# CS4234 hierarchical clustering experiments

Python and Jupyter workspace for the accompanying hierarchical clustering report.
The latest seven-algorithm comparison, including ten independent balanced-nested
graphs, is in [SELECTED_ALGORITHM_RESULTS.md](SELECTED_ALGORITHM_RESULTS.md).
Run `.venv/bin/python scripts/run_balanced_nested_replicates.py` to reproduce
the balanced-nested benchmark.
The report-ready Section 4 is in
[IMPLEMENTATION_AND_EXPERIMENTS.md](IMPLEMENTATION_AND_EXPERIMENTS.md), with five
figures, results tables, implementation details, and experimental limitations.
The current report uses Iris, Wine, and only two synthetic families: **balanced
nested** and **cross-pair stress**. Its small-instance comparison uses the six
saved Iris/Wine/balanced-nested subsets. Broader historical benchmarks remain
available in their original data and results directories.
Regenerate it from saved results using
`.venv/bin/python scripts/build_experiment_report.py`.

The additional exploratory dense-block/cross-pair family is kept separate from
the original five synthetic families. Reproduce its ten graph instances and
benchmarks with `.venv/bin/python -m hierarchical_clustering.cross_pair_experiments`.

## Setup

Python 3.13 is recommended (Python 3.11+ is supported).

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m ipykernel install --sys-prefix --name cs4234 --display-name 'Python (CS4234)'
jupyter lab notebooks/01_experiments.ipynb
```

Select **Python (CS4234)** as the notebook kernel. In VS Code, select the
`.venv/bin/python` interpreter. The editable install lets notebooks import code
from `src/` without modifying Python's search path.

## Layout

- `notebooks/01_experiments.ipynb`: experiment entry point and environment smoke check.
- `src/hierarchical_clustering/`: reusable algorithms, data handling, and metrics.
- `data/raw/`: original datasets; keep these unchanged.
- `data/processed/`: generated similarity matrices and prepared datasets.
- `results/`: experiment outputs, with plots in `results/figures/`.

Processed data, generated results, and reference PDFs are ignored by Git.
Record dataset parameters, package versions, and random seeds with experiment outputs.
Use `numpy.random.default_rng(seed)` for randomized experiments.

## Development

```bash
ruff check .
ruff format .
```

Pytest is available for tests once algorithms and metrics are implemented.
SciPy includes `scipy.optimize.linprog` for later LP-bound experiments.

## Experimental scope

Compare linkage methods, random splitting, recursive spectral cuts, and
Max-Uncut Bisection heuristics using Dasgupta cost, Moseley–Wang revenue, and
runtime. Check `cost + revenue = n * total_similarity` on each instance.
Heuristic partitioning implementations must not be assigned the guarantees of
the theoretical algorithms without establishing the required assumptions.

## Linkage experiments

```bash
.venv/bin/python -m hierarchical_clustering.experiments
.venv/bin/python -m pytest -q
```

The notebook also runs these experiments and plots cost and revenue. SciPy's
`linkage` provides single, complete, and average linkage; `to_tree` returns a
binary `ClusterNode` tree. We convert similarities to `max(W) - W` before
clustering to preserve the report's similarity merge rules. This matters for
average linkage: averaging Euclidean distances gives a different algorithm.

`results/linkage/scores.csv` contains scores and single-run tree construction
times. Six `.npy` files store the linkage matrices; load them with NumPy and
pass them to `scipy.cluster.hierarchy.to_tree` to recover tree objects.
Metadata records input hashes, library versions, and timing scope. Ties follow
SciPy's rules with original input row order. Objective evaluation uses the
original weights and leaf counts at LCAs, not dendrogram heights. No labels
are used. Scores are empirical comparisons, not ratios to the unknown optimum.

Library references:
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.to_tree.html

## Random binary tree experiments

```bash
.venv/bin/python -m hierarchical_clustering.random_experiments
```

`random_binary_tree(n, seed)` in `random_tree.py` returns a linkage-format matrix
and a SciPy tree root. Each recursive split assigns points independently with
fair coins and retries empty sides. It uses no similarities or labels and does
not force balance. Structural heights are subtree sizes, not distances.

The experiment runs seeds 4234 through 4333 on each processed dataset.
`results/random/scores.csv` records each run; `summary.csv` records means,
sample standard deviations, standard errors, and the exact expected revenue.
The two `*_trees.npz` files contain `seeds` and `trees` arrays in matching order.
Metadata records versions, input hashes, and timing scope. The notebook plots
the random revenue distribution against the linkage scores.

By symmetry, each pair has probability 1/3 of being the closest pair in any
three-leaf induced tree, so expected revenue equals `(n-2)*W/3`. This is at
least one third of optimal revenue. The guarantee is in expectation, not per
run; finite sample means may lie below it. Ratios to `(n-2)*W` use an upper
bound on OPT and are not measured approximation ratios to exact OPT.

To save a combined comparison without rerunning the algorithms:

```bash
.venv/bin/python -m hierarchical_clustering.comparison
```

This adds `revenue_upper_bound` and `revenue_over_upper_bound` to the individual
score CSVs and writes `results/revenue_comparison_sdp.csv`. Random rows report means
across seeds; linkage rows report a single tree. The denominator is `(n-2)*W`,
not exact OPT. A zero denominator is recorded as an undefined (blank) ratio.

## Recursive spectral cut

```bash
.venv/bin/python -m hierarchical_clustering.spectral_experiments
```

`spectral_tree(W)` returns a linkage-format matrix and tree root. Each recursive
step forms the induced unnormalized Laplacian with SciPy `laplacian`, obtains
the second eigenvector using `eigh`, and sweeps all nonempty prefixes to minimize
`w(A,B)/(|A||B|)`. Disconnected graphs split at whole-component boundaries,
choosing the boundary nearest half the node count. Equal sweep minima use the
first candidate. Eigenvector signs are fixed and coordinates are sorted stably;
repeated eigenvalues can still produce platform-dependent trees.

This is a heuristic without a claimed general approximation guarantee. Heights
are subtree sizes, not dissimilarities. Results, tree matrices, and metadata are
saved under `results/spectral/`, and the command updates the combined revenue
comparison. The notebook includes this method and an all-methods ratio plot.

SciPy references:
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.laplacian.html

## Recursive Max-Uncut Bisection heuristic

```bash
.venv/bin/python -m hierarchical_clustering.max_uncut_experiments
```

`max_uncut_tree(W, seed=0, rounds=32, solver_eps=1e-5)` returns a linkage-format
matrix and tree root. It solves the basic Max-Uncut SDP using CVXPY/SCS at each
nontrivial node, draws 32 Gaussian candidates, greedily balances each candidate,
and retains the one with the largest within-side weight. Uniform subgraphs have
an analytic split. Odd subgraphs are padded with one isolated dummy vertex,
then the dummy is removed to give floor(s/2) and ceil(s/2) children.

The SDP is `maximize sum(W * (1+X))/4`, subject to `X >> 0`, `diag(X)=1`,
and `X @ ones=0`. It uses the valid sign-symmetric choice `mu=0`, so Wu's
linear-bias Gaussian thresholds reduce to zero and theta has no effect.
This is a **basic-SDP heuristic inspired by Wu et al.**, not their complete
SDPH-GB algorithm: there is no Lasserre hierarchy or conditioning, and no
0.8732, 0.4246 or 0.585 approximation guarantee is claimed. `solver_eps` is
SCS numerical accuracy, not the paper's hierarchy epsilon.

Install dependencies into the project environment:

```bash
.venv/bin/python -m pip install cvxpy scs
```

The experiment runs 20 full trees per dataset (seeds 4234–4253) and records
method `max_uncut_sdp`. Scores, summaries, trees, settings and per-node solver
diagnostics are saved in `results/max_uncut_sdp/`. The new combined table is
`results/revenue_comparison_sdp.csv`. Earlier NetworkX results remain in their
original folders as historical data and are excluded from the new comparison.

The SDP primal objective is a numerical estimate, not a certified upper bound.
Small PSD/diagonal errors are repaired for Gaussian sampling; materially
infeasible solver output raises an error. Repair may perturb moment balance,
so every sampled partition is explicitly balanced afterward. Tree heights are
leaf counts, not distances. Only similarity weights are used.

References: Wu et al., `max-uncut.pdf`, Algorithm 1 Steps 3.2–4;
https://www.cvxpy.org/examples/basic/sdp.html

## Root-bisection constructions

The root-only variants are separate from recursive Max-Uncut:

```bash
.venv/bin/python -m hierarchical_clustering.hybrid_experiments
```

This runs all seven datasets with 20 seeds and 32 Gaussian candidates per root
partition. One basic SDP is solved per dataset and reused across seeds. For each
seed, both constructions use the same balanced root partition:

- `mub_average_candidate`: average linkage separately inside the two halves.
- `mub_average_best_of`: return the better-revenue tree between that candidate
  and global average linkage, as required by the Ahmadian et al. construction.
  Global average linkage wins ties. This retains average linkage's 1/3 guarantee.
- `mub_random`: independent fair-coin recursive splits inside the two halves,
  following the Alon--Azar--Vainstein outer construction. Only the root is forced
  balanced; internal random splits are conditioned only on being nonempty.

Both use our basic-SDP heuristic as the root subroutine, so we do **not** claim
the papers' 0.4246 or 0.585 guarantees. The raw average candidate is saved to
distinguish the benefit of bisection from the protection provided by best-of.

Completed results for all seven datasets are in
[ROOT_BISECTION_RESULTS.md](ROOT_BISECTION_RESULTS.md).

Tiny-instance hierarchy and conditioning checks are now available separately:

```bash
.venv/bin/python -m hierarchical_clustering.small_hierarchy_experiments
```

This tests 6- and 8-vertex subsets and diagnostic graphs against exact bisection
and tree optima. It solves actual level-1/2 moment SDPs for comparison and uses
the full hierarchy's equivalent probability LP for conditioning and Wu's
linear-bias rounding. The full-level calculation is exponential and deliberately
restricted to tiny graphs. Expected rounding quality is enumerated, not inferred
from a small sample of seeds. See [SMALL_HIERARCHY_RESULTS.md](SMALL_HIERARCHY_RESULTS.md)
and `notebooks/04_small_hierarchy_checks.ipynb` for results and the finite-size
qualifications needed before quoting approximation factors.

Compare all active algorithms on those same tiny inputs using
`.venv/bin/python -m hierarchical_clustering.small_comparison`.
[SMALL_ALGORITHM_COMPARISON.md](SMALL_ALGORITHM_COMPARISON.md) reports revenue
against exact OPT, including the distinct expectation calculations and rounding
budgets. Results and tree archives are in `results/small_algorithm_comparison/`.

Use `max_uncut_hybrid_trees(W, seed=4234)` from
`hierarchical_clustering.max_uncut_hybrids` for a single run. It returns a mapping
from method names to linkage matrices and root/selection diagnostics.

Output goes to `results/root_bisection/`: scores, summary, ratio comparison,
tree archives, root partitions, solver diagnostics and input hashes. The random
method also records exact conditional expected revenue given its root partition;
its observed revenue remains the scored value of the actual sampled tree.
Shared solve/round/build times are explicitly identified, not compared as
independent per-method timings. Open `notebooks/03_root_bisection_variants.ipynb`
for the side-by-side comparisons. Existing recursive-SDP results are retained.

References: https://proceedings.mlr.press/v108/chatziafratis20a/chatziafratis20a.pdf
and https://proceedings.mlr.press/v125/alon20b/alon20b.pdf.

## Synthetic families

```bash
.venv/bin/python -m hierarchical_clustering.synthetic
```

This generates five 128-point similarity graphs (seed 4234, noise std 0.02)
under `data/processed/synthetic/`. Open `notebooks/02_synthetic_datasets.ipynb`
to regenerate, inspect, and plot them. Each `.npz` contains `similarities`,
`planted_similarities` (before noise), `labels`, and ordering information.
Family-specific arrays include hierarchical group IDs, mixed memberships,
core/periphery roles, or bridge positions. JSON metadata and a manifest record
all parameters. Load with `np.load(path, allow_pickle=False)`.

The configurable generators in `synthetic.py` cover:

| Family | Default structure | Main controls |
|---|---|---|
| `balanced_nested` | 3 binary levels, 8 equal leaf groups | `depth`, `low`, `high` |
| `unbalanced_nested` | 3 levels, 75:25 splits where feasible | `depth`, `split_fraction`, `low`, `high` |
| `chains_bridges` | 4 groups joined by 3 six-point chains | `groups`, `bridge_length`, `within`, `background`, `bridge_weight` |
| `overlapping` | 4 groups, 25% mixed points | `groups`, `overlap_fraction`, `background` |
| `core_periphery` | 20% core and 4 peripheral groups | `groups`, `core_fraction`, `core_within`, `periphery_within`, `core_to_periphery`, `between_peripheries` |

All accept `n`, `seed`, `noise`, and `shuffle`. Nested similarities grow linearly
with the number of shared hierarchy levels. Balanced splits differ by at most
one point when sizes are odd. Unbalanced splits reserve enough points to keep
all requested descendant groups nonempty. Overlap affinities use cosine similarity
of membership vectors, with mixed points assigned equally to two distinct groups.
Chain endpoints connect strongly to the adjacent dense group; other nonlocal
connections use the background weight. The core connects to all peripheral groups.

Noise is independent Gaussian noise on unordered pairs, clipped to [0,1].
Graphs are symmetric with zero diagonals. Construction, noise, and permutation
use separate seeded random streams. `original_indices` maps saved rows to the
construction order; `display_order` restores that order for heatmaps. Labels
`-1` denote bridge or mixed points with no unique group membership.

These are direct graph inputs: do not standardize them or apply the Gaussian
feature preprocessing. Planted labels are only for evaluation, and are not a
proof of optimal hierarchy. The original experiment runners target Iris and Wine;
the synthetic runner below evaluates the saved synthetic graphs.

### Benchmark synthetic examples

```bash
.venv/bin/python -m hierarchical_clustering.synthetic_experiments
```

All six methods receive the same saved weights per family. Single, complete,
average, and spectral run once; random trees use seeds 4234–4333 and Max-Uncut
uses seeds 4234–4253 (32 Gaussian candidates per node). This produces
620 trees across the five graphs. No labels or planted reference matrices enter
the algorithms. New basic-SDP output is under `results/synthetic_sdp/`:

- `scores.csv`: individual scores, timings, seeds, and revenue-to-upper-bound ratios.
- `summary.csv`: means, sample standard deviations, and standard errors.
- `revenue_ratio_comparison.csv`: family-by-method ratio table.
- `*_trees.npz`: trees, methods, and algorithm seeds in matching order.
- `metadata.json`: generation settings, input hashes, algorithm settings, and versions.

The synthetic notebook loads these results and plots ratios with standard-deviation
error bars. Rerun the benchmark after changing the saved graph settings. There is
only one generated instance per family; variation reflects algorithm seeds,
not variability across datasets. Timings are preliminary, not controlled repeated
benchmarks of the deterministic methods. Ratios use an upper bound, not exact OPT.

The completed basic-SDP experiment is summarized in
[SDP_EXPERIMENT_RESULTS.md](SDP_EXPERIMENT_RESULTS.md). The 140 SDP trees
(20 per dataset) and all synthetic baseline trees were checked against their
saved scores and source hashes. Per-node SCS diagnostics are saved with the runs.
# cs4234
