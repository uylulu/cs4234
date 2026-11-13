# UCI datasets

Retrieved on 2026-11-11 from the official UCI text files:

- `iris.csv`: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
- `wine.csv`: https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data

Shell DNS was unavailable, so these files were saved from the web retrieval
tool's line-numbered view of the official sources. All source row indices were
present and overlapping rows agreed. Data values and row order are preserved;
blank lines and file-ending whitespace are normalized. Neither CSV has a header.

Iris contains 150 rows: four numerical features followed by the species label.
Wine contains 178 rows: the class label followed by 13 numerical features.
This is Wine, not Wine Quality. Dimensions, finite feature values, and class
counts were checked (Iris: 50/50/50; Wine: 59/71/48).

Keep these files as raw data. For experiments, separate the labels, standardize
only numerical features, then calculate Euclidean distances and Gaussian
similarities. Labels must not enter the similarity calculation. Save derived
data under `processed/`. Choose and record the Gaussian bandwidth in the
experiment protocol; set the similarity diagonal to zero for graph objectives.

## Processed artifacts

The selected bandwidth is the median strictly positive Euclidean distance over
unordered pairs after standardization, separately for each dataset.

- `processed/iris.npz` and `processed/wine.npz`: standardized features, labels,
  Euclidean distances, Gaussian similarities, feature means/scales, and sigma.
- `processed/iris_metadata.json` and `processed/wine_metadata.json`: parameters,
  source hashes, class counts, and NumPy version.

Rows retain their original order. Labels are separate string arrays; no labels
enter scaling or similarity calculations. Load archives with
`np.load(path, allow_pickle=False)` and use the `similarities` array as graph input.

Reproduce from the project root with:

```bash
PYTHONPATH=src python3 -m hierarchical_clustering.preprocessing
```

This command needs NumPy, SciPy, and scikit-learn (declared in `pyproject.toml`).
Scaling uses `StandardScaler`, with population standard deviations (`ddof=0`).
SciPy's `pdist` computes unordered pairwise distances and `squareform` builds the
distance matrix. The notebook also includes cells to regenerate and load these artifacts.

## Exploratory cross-pair graphs

`processed/exploratory_cross_pairs/` contains ten additional 32-vertex graphs.
Each has two equally sized dense groups (within weight 0.6), a stronger matching
across groups (weight 1), and other cross-group weights 0.02. Independent pair
noise has standard deviation 0.01, followed by clipping to [0,1] and shuffling.
Generation seeds are 5000–5009. These directly specified similarity graphs form
a deliberately constructed stress test, separate from the five original families.
Reproduce them and their benchmarks with:

```bash
.venv/bin/python -m hierarchical_clustering.cross_pair_experiments
```

Settings, source hashes, and scores are under `results/exploratory_cross_pairs/`.
