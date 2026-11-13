"""Recursive basic-SDP Max-Uncut heuristic with Wu-style rounding.

No Lasserre hierarchy, conditioning, or 0.8732 guarantee is implemented.
The sign-symmetric basic SDP admits mu=0, so theta is inactive here.
"""

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import to_tree

from hierarchical_clustering.linkage import validate_similarities
from hierarchical_clustering.wu_rounding import gaussian_rounding, greedy_balance

METHOD = "max_uncut_sdp"


@dataclass
class SDPSolution:
    correlations: np.ndarray
    diagnostics: dict


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def solve_basic_sdp(similarities, *, solver_eps=1e-5, solver_max_iters=20_000):
    """Solve max sum_{i<j} w_ij (1+X_ij)/2, X PSD, diag(X)=1, X1=0.

    CVXPY models the convex SDP and SCS solves it. We report feasibility
    residuals, reject materially infeasible solutions, and repair small PSD/
    diagonal errors for sampling. The floating-point primal objective is an
    estimate, NOT a certified upper bound. Repair need not preserve X1=0;
    exact partition balance is restored by greedy rounding.
    """
    w = validate_similarities(similarities)
    n = len(w)
    if n % 2:
        raise ValueError("The basic bisection SDP requires even n; pad odd graphs first")
    if not np.isfinite(solver_eps) or not 0 < solver_eps <= 1e-3:
        raise ValueError("solver_eps must be finite and in (0, 1e-3]")
    _positive_int("solver_max_iters", solver_max_iters)
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            "Install SDP dependencies: .venv/bin/python -m pip install cvxpy scs"
        ) from exc

    if not np.any(w):
        x = (n * np.eye(n) - np.ones((n, n))) / (n - 1)
        return SDPSolution(x, {"status": "analytic_zero", "n_sdp": n})
    x_var = cp.Variable((n, n), symmetric=True)
    scale = float(w.max())
    problem = cp.Problem(
        cp.Maximize(cp.sum(cp.multiply(w / scale, 1 + x_var)) / 4),
        [x_var >> 0, cp.diag(x_var) == 1, x_var @ np.ones(n) == 0],
    )
    problem.solve(solver="SCS", eps=solver_eps, max_iters=solver_max_iters)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or x_var.value is None:
        raise RuntimeError(f"Basic SDP failed: SCS status={problem.status}")
    raw = np.asarray(x_var.value)
    if not np.isfinite(raw).all():
        raise RuntimeError("Basic SDP returned nonfinite moments")
    raw = (raw + raw.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(raw)
    diagonal_error = float(np.max(np.abs(np.diag(raw) - 1)))
    balance_error = float(np.max(np.abs(raw.sum(axis=1))) / n)
    tolerance = max(1e-6, 20 * solver_eps)
    if min(eigenvalues) < -tolerance or max(diagonal_error, balance_error) > tolerance:
        raise RuntimeError(
            "Basic SDP failed feasibility checks: "
            f"min_eigenvalue={min(eigenvalues):.3g}, diag={diagonal_error:.3g}, "
            f"normalized_balance={balance_error:.3g}; increase solver_max_iters"
        )
    factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
    factor /= np.linalg.norm(factor, axis=1, keepdims=True)
    repaired = factor @ factor.T
    stats = problem.solver_stats
    diagnostics = {
        "status": problem.status,
        "n_sdp": n,
        "sdp_objective_estimate": float(np.sum(w * (1 + raw)) / 4),
        "min_eigenvalue": float(eigenvalues.min()),
        "diagonal_error": diagonal_error,
        "normalized_balance_error": balance_error,
        "repair_max_entry_change": float(np.max(np.abs(repaired - raw))),
        "repaired_normalized_balance_error": float(np.max(np.abs(repaired.sum(axis=1))) / n),
        "solver_iterations": stats.num_iters,
        "solver_seconds": stats.solve_time,
    }
    return SDPSolution(repaired, diagnostics)


def _balanced_cut(w, rng, rounds, solver_eps, solver_max_iters):
    n = len(w)
    # Removing a zero-weight dummy from an equal bisection gives floor/ceil sizes.
    padded = np.pad(w, ((0, n % 2), (0, n % 2)))
    solution = solve_basic_sdp(padded, solver_eps=solver_eps, solver_max_iters=solver_max_iters)
    return round_basic_sdp(w, solution, seed=rng, rounds=rounds)


def round_basic_sdp(similarities, solution, *, seed=0, rounds=32):
    """Round a previously solved SDP for this graph (padded if odd).

    Reusing the same deterministic SDP solve across seeds avoids redundant
    optimization. The caller must supply the solution for these exact weights.
    Returns the best partition and a fresh diagnostics dictionary.
    """
    w = validate_similarities(similarities)
    _positive_int("rounds", rounds)
    n = len(w)
    padded = np.pad(w, ((0, n % 2), (0, n % 2)))
    if solution.correlations.shape != padded.shape:
        raise ValueError("SDP solution has the wrong shape for the padded graph")
    assignments = gaussian_rounding(
        np.zeros(len(padded)), solution.correlations, samples=rounds, seed=seed
    )
    best_partition, best_cut = None, np.inf
    for signs in assignments:
        signs = greedy_balance(padded, signs)[:n]
        a, b = np.flatnonzero(signs == 1), np.flatnonzero(signs == -1)
        cut = float(w[np.ix_(a, b)].sum())
        if cut < best_cut:
            best_partition, best_cut = (a, b), cut
    diagnostics = dict(solution.diagnostics)
    diagnostics.update(
        n_original=n,
        dummy_vertex=bool(n % 2),
        rounded_uncut=float(w.sum() / 2 - best_cut),
    )
    return best_partition, diagnostics


def max_uncut_tree(
    similarities,
    seed=0,
    *,
    rounds=32,
    solver_eps=1e-5,
    solver_max_iters=20_000,
    return_diagnostics=False,
):
    """Return linkage matrix and root using basic SDP at every nontrivial split.

    Optional third return value contains per-node numerical diagnostics. The
    tree is a heuristic: no 0.8732, 0.4246 or 0.585 guarantee is claimed.
    ``solver_eps`` is SCS accuracy, NOT the paper's hierarchy epsilon.
    """
    w = validate_similarities(similarities)
    _positive_int("rounds", rounds)
    _positive_int("solver_max_iters", solver_max_iters)
    if not np.isfinite(solver_eps) or not 0 < solver_eps <= 1e-3:
        raise ValueError("solver_eps must be finite and in (0, 1e-3]")
    rng = np.random.default_rng(seed)
    n, rows, diagnostics = len(w), [], []

    def split(leaves):
        if len(leaves) == 1:
            return int(leaves[0])
        local = w[np.ix_(leaves, leaves)]
        edges = local[np.triu_indices(len(leaves), 1)]
        if len(leaves) == 2 or np.all(edges == edges[0]):
            a, b = np.array_split(leaves, 2)
            info = {"status": "analytic_uniform", "n_original": len(leaves)}
        else:
            (i, j), info = _balanced_cut(local, rng, rounds, solver_eps, solver_max_iters)
            a, b = leaves[i], leaves[j]
        info["leaves"] = leaves.tolist()
        diagnostics.append(info)
        left, right = split(a), split(b)
        node_id = n + len(rows)
        rows.append([left, right, len(leaves), len(leaves)])
        return node_id

    split(np.arange(n))
    merges = np.asarray(rows, dtype=float)
    result = (merges, to_tree(merges))
    return (*result, diagnostics) if return_diagnostics else result


def experiment_settings(rounds, solver_eps, solver_max_iters):
    """Shared provenance for real and synthetic experiment runners."""
    import cvxpy
    import scs

    return {
        "method": METHOD,
        "cvxpy_version": cvxpy.__version__,
        "scs_version": scs.__version__,
        "solver": "SCS",
        "solver_eps": solver_eps,
        "solver_max_iters": solver_max_iters,
        "rounds_per_node": rounds,
        "relaxation": "basic SDP: X PSD, diag(X)=1, X @ ones=0",
        "rounding": "Wu-style Gaussian thresholds with mu=0; theta inactive",
        "balancing": "greedy minimum uncut loss; odd nodes padded with zero-weight dummy",
        "numerical_repair": "clip small negative eigenvalues; normalize diagonal for sampling",
        "guarantee": "heuristic; no hierarchy, conditioning, or claimed approximation factor",
        "sdp_objective": "floating-point primal estimate, not a certified upper bound",
    }
