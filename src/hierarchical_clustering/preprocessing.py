"""Prepare UCI features and median-bandwidth Gaussian similarity graphs.

Run from the project root:
    PYTHONPATH=src python3 -m hierarchical_clustering.preprocessing
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler


def gaussian_graph(features):
    """Standardize features and return distances, similarities, and fitted parameters.

    StandardScaler uses population variance (ddof=0) and handles constant
    features. Pass numerical features only, with labels removed.
    """

    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] == 0:
        raise ValueError("Expected at least two samples and one numerical feature")
    if not np.isfinite(x).all():
        raise ValueError("Features contain missing or nonfinite values")

    scaler = StandardScaler()
    z = scaler.fit_transform(x)

    # pdist returns each unordered pair once, excluding the diagonal.
    pairs = pdist(z, metric="euclidean")
    positive = pairs[pairs > 0]
    if not len(positive):
        raise ValueError("No nonzero pairwise distances; median bandwidth is undefined")
    sigma = float(np.median(positive))

    distances = squareform(pairs)
    similarities = np.exp(-0.5 * (distances / sigma) ** 2)
    np.fill_diagonal(similarities, 0.0)
    if not (
        np.isfinite(similarities).all()
        and np.allclose(similarities, similarities.T)
        and np.all((similarities >= 0) & (similarities <= 1))
        and np.all(np.diag(similarities) == 0)
    ):
        raise ValueError("Invalid similarity matrix")
    return z, distances, similarities, scaler.mean_, scaler.scale_, sigma


def preprocess_dataset(name, data_dir):
    """Validate one original UCI CSV and save arrays and provenance metadata."""
    specs = {"iris": (150, 4, 4), "wine": (178, 13, 0)}
    if name not in specs:
        raise ValueError(f"Unknown dataset: {name}")

    n, feature_count, label_column = specs[name]
    data_dir = Path(data_dir)
    source = data_dir / f"{name}.csv"
    with source.open(newline="") as stream:
        rows = [row for row in csv.reader(stream) if row]
    if len(rows) != n or any(len(row) != feature_count + 1 for row in rows):
        raise ValueError(f"Unexpected dimensions for {name}")
    labels = np.array([row[label_column] for row in rows])
    expected_labels = (
        {"Iris-setosa", "Iris-versicolor", "Iris-virginica"}
        if name == "iris" else {"1", "2", "3"}
    )
    if set(labels) != expected_labels:
        raise ValueError(f"Unexpected labels for {name}")
    features = np.array([
        [float(value) for j, value in enumerate(row) if j != label_column]
        for row in rows
    ])
    z, distances, similarities, mean, scale, sigma = gaussian_graph(features)
    output = data_dir / "processed"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / f"{name}.npz",
        features_standardized=z,
        labels=labels,
        distances=distances,
        similarities=similarities,
        mean=mean,
        scale=scale,
        sigma=np.array(sigma),
    )
    unique, counts = np.unique(labels, return_counts=True)
    metadata = {
        "dataset": name,
        "source": f"https://archive.ics.uci.edu/ml/machine-learning-databases/{name}/{name}.data",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "samples": n,
        "features": feature_count,
        "label_column_zero_based": label_column,
        "class_counts": dict(zip(unique.tolist(), counts.tolist())),
        "scaling": "(x - mean) / population_std; ddof=0; constant feature scale=1",
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "bandwidth_rule": "median of strictly positive distances over i < j",
        "sigma": sigma,
        "similarity": "exp(-distance**2 / (2 * sigma**2)); diagonal=0",
        "row_order": "unchanged from source CSV",
        "numpy_version": np.__version__,
    }
    (output / f"{name}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    for dataset in ("iris", "wine"):
        info = preprocess_dataset(dataset, root / "data")
        print(f"{dataset}: {info['samples']} samples, sigma={info['sigma']:.8f}")
