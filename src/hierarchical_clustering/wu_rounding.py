"""Wu--Du--Xu linear-bias rounding and greedy size adjustment.

These implement the rounding kernel in Algorithm SDPH-GB, Steps 3.2--4.
They are NOT the full 0.8732-approximation algorithm: the caller must supply
the vectors from the Lasserre solve, conditioning and Step 3.1 modification.
Accepting an arbitrary valid Gram matrix does not certify the approximation.
"""

import numpy as np
from scipy.special import ndtri

from hierarchical_clustering.linkage import validate_similarities


def gaussian_rounding(mu, correlations, *, theta=0.87, samples=1, seed=0):
    """Return sign assignments from Wu's correlated Gaussian threshold rule.

    ``mu[i] = <v0, vi>`` and ``correlations[i,j] = <vi,vj>``. Zero residual
    vectors receive mutually orthogonal directions, as in Step 3.2. This
    function validates moment geometry but cannot validate conditioning history.
    ``seed`` can also be a NumPy Generator, allowing one RNG per experiment.
    """
    mu = np.asarray(mu, dtype=float)
    rho = np.asarray(correlations, dtype=float)
    if mu.ndim != 1 or len(mu) < 2 or rho.shape != (len(mu), len(mu)):
        raise ValueError("Expected a vector mu and a matching square correlation matrix")
    if not np.isfinite(mu).all() or not np.isfinite(rho).all():
        raise ValueError("Moments must be finite")
    if not np.isfinite(theta) or not 0 <= theta <= 1:
        raise ValueError("theta must lie in [0, 1]")
    if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)) or samples < 1:
        raise ValueError("samples must be a positive integer")
    tol = 1e-8
    if np.any(np.abs(mu) > 1 + tol) or abs(mu.sum()) > tol * len(mu):
        raise ValueError("mu must lie in [-1, 1] and sum to zero")
    if not np.allclose(rho, rho.T, atol=tol, rtol=0):
        raise ValueError("Correlation matrix must be symmetric")
    if not np.allclose(np.diag(rho), 1, atol=tol, rtol=0):
        raise ValueError("Correlation matrix must have unit diagonal")
    mu = np.clip(mu, -1, 1)
    rho = (rho + rho.T) / 2
    residual = rho - np.outer(mu, mu)
    if np.linalg.eigvalsh(residual).min() < -tol:
        raise ValueError("Moments do not form a positive semidefinite Gram matrix")
    lengths = np.sqrt(np.maximum(1 - mu**2, 0))
    active = lengths > 1e-12
    if np.any(np.abs(residual[~active]) > tol):
        raise ValueError("Zero residual vectors must have zero residual covariances")
    covariance = np.eye(len(mu))
    covariance[np.ix_(active, active)] = (
        residual[np.ix_(active, active)] / np.outer(lengths[active], lengths[active])
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues.min() < -tol:
        raise ValueError("Normalized residual covariance is not positive semidefinite")
    factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
    # Remove only floating-point drift from unit Gaussian marginal variances.
    factor /= np.linalg.norm(factor, axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    projections = rng.standard_normal((samples, len(mu))) @ factor.T
    thresholds = ndtri((1 - theta * mu) / 2)
    return np.where(projections < thresholds, -1, 1)


def greedy_balance(similarities, assignment):
    """Balance an even graph by moving the vertex with smallest uncut loss.

    At each step only vertices on the larger side are eligible. This greedy
    realization satisfies Wu's Lemma 1 uncut retention bound: for initial
    larger-side size m and target k=n/2, final >= k(k-1)/(m(m-1))*initial.
    No Kernighan--Lin swaps or additional local optimization are performed.
    """
    w = validate_similarities(similarities)
    if len(w) % 2:
        raise ValueError("Wu's bisection kernel requires an even number of vertices")
    x = np.asarray(assignment)
    if x.shape != (len(w),) or not np.isin(x, [-1, 1]).all():
        raise ValueError("assignment must contain one -1/+1 sign per vertex")
    x = x.astype(int, copy=True)
    while x.sum() != 0:
        majority = 1 if x.sum() > 0 else -1
        candidates = np.flatnonzero(x == majority)
        # Flipping i loses its same-side weight and gains its opposite-side weight.
        losses = x[candidates] * (w[candidates] @ x)
        x[candidates[np.argmin(losses)]] *= -1
    return x
