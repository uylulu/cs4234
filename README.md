# CS4234 hierarchical clustering experiments

Reproducible implementations and results for seven hierarchical clustering methods on Iris, Wine, and two synthetic weighted graph families. See [IMPLEMENTATION_AND_EXPERIMENTS.md](IMPLEMENTATION_AND_EXPERIMENTS.md) for the report section and [SELECTED_ALGORITHM_RESULTS.md](SELECTED_ALGORITHM_RESULTS.md) for detailed scores and plots.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Raw Iris and Wine files are in `data/`. Preprocessing standardizes numerical features, computes Euclidean distances, then applies a Gaussian kernel with median positive distance bandwidth. Labels are excluded.

## Reproduce

Run these scripts in order:

```bash
.venv/bin/python scripts/run_balanced_nested_replicates.py
.venv/bin/python scripts/selected_algorithm_results.py
```

- `run_balanced_nested_replicates.py` generates ten balanced nested graphs with
  seeds 4234–4243, runs the seven algorithms, and saves scores and summaries.
- `selected_algorithm_results.py` runs the Max-Uncut variants on Iris, Wine,
  and ten cross-pair graphs (seeds 5000–5009), combines those scores with the
  linkage, random-tree, spectral, and balanced-nested results, then regenerates
  the result tables and comparison plot in `SELECTED_ALGORITHM_RESULTS.md` and
  `results/selected_algorithms/`.

Generated similarity matrices, score files, tree archives, and plots are kept
locally and ignored by Git. The raw cross-pair graphs use seeds 5000–5009.

The Max-Uncut methods use a basic SDP relaxation and do not inherit the theoretical guarantees of the full SDP hierarchy.
