"""
Core computational utilities for VIP-OT.

This module provides a compact, standalone implementation of the main
optimal-transport operations used throughout the VIP-OT analyses:

- pairwise cost-matrix construction
- exact balanced optimal transport (EMD)
- entropically regularized Sinkhorn transport
- row normalization of transport plans
- nearest-neighbor, Hungarian, and mutual-nearest-neighbor matching
- barycentric target estimation
- single-cell perturbation-vector construction
- mean difference feature spectra (DFSP)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
import ot
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


ArrayLike = np.ndarray


@dataclass
class TransportResult:
    """Container for a soft transport result."""

    cost_matrix: np.ndarray
    raw_coupling: np.ndarray
    row_coupling: np.ndarray


@dataclass
class PerturbationResult:
    """Container for source-level VIP-OT perturbation outputs."""

    barycentric_target: np.ndarray
    perturbation_vectors: np.ndarray
    mean_dfsp: np.ndarray


def validate_spectral_matrices(
    source: ArrayLike,
    target: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and convert source and target spectral matrices.

    Parameters
    ----------
    source
        Source-cell spectra with shape (n_source, n_features).
    target
        Target-cell spectra with shape (n_target, n_features).

    Returns
    -------
    source_arr, target_arr
        Float64 NumPy arrays.

    Raises
    ------
    ValueError
        If the matrices are not two-dimensional, have incompatible feature
        dimensions, contain zero cells, or include NaN/Inf values.
    """
    source_arr = np.asarray(source, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)

    if source_arr.ndim != 2 or target_arr.ndim != 2:
        raise ValueError("source and target must both be two-dimensional arrays.")

    if source_arr.shape[0] == 0 or target_arr.shape[0] == 0:
        raise ValueError("source and target must each contain at least one cell.")

    if source_arr.shape[1] != target_arr.shape[1]:
        raise ValueError(
            "source and target must have the same number of spectral features."
        )

    if not np.isfinite(source_arr).all():
        raise ValueError("source contains NaN or Inf values.")

    if not np.isfinite(target_arr).all():
        raise ValueError("target contains NaN or Inf values.")

    return source_arr, target_arr


def compute_cost_matrix(
    source: ArrayLike,
    target: ArrayLike,
    metric: str = "sqeuclidean",
) -> np.ndarray:
    """
    Compute the pairwise source-to-target cost matrix.

    Parameters
    ----------
    source, target
        Spectral matrices with shape (n_cells, n_features).
    metric
        Distance metric accepted by scipy.spatial.distance.cdist.
        VIP-OT analyses use squared Euclidean distance by default.

    Returns
    -------
    cost_matrix
        Array with shape (n_source, n_target).
    """
    source_arr, target_arr = validate_spectral_matrices(source, target)
    return cdist(source_arr, target_arr, metric=metric)


def row_normalize_coupling(
    coupling: ArrayLike,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    Convert a transport plan into source-conditioned weights.

    Each row is normalized to sum to one. The resulting matrix can therefore
    be interpreted as the conditional distribution over target cells for each
    source cell.

    Parameters
    ----------
    coupling
        Nonnegative transport matrix with shape (n_source, n_target).
    eps
        Numerical threshold used to detect empty rows.

    Returns
    -------
    normalized
        Row-normalized transport matrix.

    Raises
    ------
    ValueError
        If the coupling is invalid or contains an empty source row.
    """
    coupling_arr = np.asarray(coupling, dtype=np.float64)

    if coupling_arr.ndim != 2:
        raise ValueError("coupling must be a two-dimensional array.")

    if not np.isfinite(coupling_arr).all():
        raise ValueError("coupling contains NaN or Inf values.")

    if np.any(coupling_arr < -eps):
        raise ValueError("coupling contains negative transport weights.")

    coupling_arr = np.clip(coupling_arr, 0.0, None)
    row_sums = coupling_arr.sum(axis=1, keepdims=True)

    if np.any(row_sums <= eps):
        empty_rows = np.flatnonzero(row_sums.ravel() <= eps)
        raise ValueError(
            f"coupling contains {len(empty_rows)} empty source row(s): "
            f"{empty_rows[:10].tolist()}"
        )

    return coupling_arr / row_sums


def compute_emd_coupling(
    source: ArrayLike,
    target: ArrayLike,
    metric: str = "sqeuclidean",
) -> TransportResult:
    """
    Compute exact balanced optimal transport using Earth Mover's Distance.

    Uniform source and target marginals are used, followed by row
    normalization for source-conditioned barycentric analysis.

    Parameters
    ----------
    source, target
        Spectral matrices with shape (n_cells, n_features).
    metric
        Distance metric used to construct the cost matrix.

    Returns
    -------
    TransportResult
        Cost matrix, raw balanced transport plan, and row-normalized coupling.
    """
    source_arr, target_arr = validate_spectral_matrices(source, target)
    cost_matrix = compute_cost_matrix(source_arr, target_arr, metric=metric)

    source_mass = np.full(
        source_arr.shape[0],
        1.0 / source_arr.shape[0],
        dtype=np.float64,
    )
    target_mass = np.full(
        target_arr.shape[0],
        1.0 / target_arr.shape[0],
        dtype=np.float64,
    )

    raw_coupling = ot.emd(source_mass, target_mass, cost_matrix)
    row_coupling = row_normalize_coupling(raw_coupling)

    return TransportResult(
        cost_matrix=cost_matrix,
        raw_coupling=raw_coupling,
        row_coupling=row_coupling,
    )


def compute_sinkhorn_coupling(
    source: ArrayLike,
    target: ArrayLike,
    reg: float = 0.05,
    metric: str = "sqeuclidean",
    num_iter_max: int = 100_000,
    stop_thr: float = 1e-9,
) -> TransportResult:
    """
    Compute entropically regularized balanced optimal transport.

    Parameters
    ----------
    source, target
        Spectral matrices with shape (n_cells, n_features).
    reg
        Entropic regularization strength.
    metric
        Distance metric used to construct the cost matrix.
    num_iter_max
        Maximum number of Sinkhorn iterations.
    stop_thr
        Sinkhorn convergence tolerance.

    Returns
    -------
    TransportResult
        Cost matrix, raw transport plan, and row-normalized coupling.
    """
    if reg <= 0:
        raise ValueError("reg must be greater than zero.")

    source_arr, target_arr = validate_spectral_matrices(source, target)
    cost_matrix = compute_cost_matrix(source_arr, target_arr, metric=metric)

    source_mass = np.full(
        source_arr.shape[0],
        1.0 / source_arr.shape[0],
        dtype=np.float64,
    )
    target_mass = np.full(
        target_arr.shape[0],
        1.0 / target_arr.shape[0],
        dtype=np.float64,
    )

    raw_coupling = ot.sinkhorn(
        source_mass,
        target_mass,
        cost_matrix,
        reg=reg,
        numItermax=num_iter_max,
        stopThr=stop_thr,
    )
    row_coupling = row_normalize_coupling(raw_coupling)

    return TransportResult(
        cost_matrix=cost_matrix,
        raw_coupling=raw_coupling,
        row_coupling=row_coupling,
    )


def compute_barycentric_target(
    target: ArrayLike,
    row_coupling: ArrayLike,
) -> np.ndarray:
    """
    Compute the expected target spectrum for every source cell.

    Parameters
    ----------
    target
        Target-cell spectral matrix with shape (n_target, n_features).
    row_coupling
        Source-conditioned coupling with shape (n_source, n_target).

    Returns
    -------
    barycentric_target
        Matrix with shape (n_source, n_features).
    """
    target_arr = np.asarray(target, dtype=np.float64)
    coupling_arr = np.asarray(row_coupling, dtype=np.float64)

    if target_arr.ndim != 2 or coupling_arr.ndim != 2:
        raise ValueError("target and row_coupling must both be two-dimensional.")

    if coupling_arr.shape[1] != target_arr.shape[0]:
        raise ValueError(
            "row_coupling column count must equal the number of target cells."
        )

    normalized = row_normalize_coupling(coupling_arr)
    return normalized @ target_arr


def compute_perturbation_vectors(
    source: ArrayLike,
    target: ArrayLike,
    row_coupling: ArrayLike,
) -> PerturbationResult:
    """
    Compute source-level barycentric targets and perturbation vectors.

    The perturbation vector for source cell i is:

        Delta_i = sum_j gamma_ij * target_j - source_i

    where gamma is row-normalized.

    Parameters
    ----------
    source, target
        Source and target spectral matrices.
    row_coupling
        Row-normalized source-to-target coupling.

    Returns
    -------
    PerturbationResult
        Barycentric targets, cell-level perturbation vectors, and the mean
        difference feature spectrum.
    """
    source_arr, target_arr = validate_spectral_matrices(source, target)
    coupling_arr = np.asarray(row_coupling, dtype=np.float64)

    if coupling_arr.shape != (source_arr.shape[0], target_arr.shape[0]):
        raise ValueError(
            "row_coupling must have shape "
            f"({source_arr.shape[0]}, {target_arr.shape[0]})."
        )

    barycentric_target = compute_barycentric_target(target_arr, coupling_arr)
    perturbation_vectors = barycentric_target - source_arr
    mean_dfsp = perturbation_vectors.mean(axis=0)

    return PerturbationResult(
        barycentric_target=barycentric_target,
        perturbation_vectors=perturbation_vectors,
        mean_dfsp=mean_dfsp,
    )


def compute_edge_differences(
    source: ArrayLike,
    target: ArrayLike,
    row_coupling: ArrayLike,
    threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract nonzero transport edges and their target-minus-source differences.

    Parameters
    ----------
    source, target
        Source and target spectral matrices.
    row_coupling
        Row-normalized coupling.
    threshold
        Retain transport edges with weight strictly greater than this value.

    Returns
    -------
    source_indices, target_indices, weights, differences
        Sparse edge representation and edge-level spectral differences.
    """
    source_arr, target_arr = validate_spectral_matrices(source, target)
    coupling_arr = row_normalize_coupling(row_coupling)

    if coupling_arr.shape != (source_arr.shape[0], target_arr.shape[0]):
        raise ValueError("row_coupling shape is incompatible with source and target.")

    source_idx, target_idx = np.where(coupling_arr > threshold)
    weights = coupling_arr[source_idx, target_idx]
    differences = target_arr[target_idx] - source_arr[source_idx]

    return source_idx, target_idx, weights, differences


def match_nearest_neighbor(
    source: ArrayLike,
    target: ArrayLike,
    metric: str = "sqeuclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy nearest-neighbor matching from each source cell to one target cell.

    Target reuse is allowed.
    """
    cost_matrix = compute_cost_matrix(source, target, metric=metric)
    source_idx = np.arange(cost_matrix.shape[0], dtype=int)
    target_idx = np.argmin(cost_matrix, axis=1).astype(int)
    return source_idx, target_idx


def match_hungarian(
    source: ArrayLike,
    target: ArrayLike,
    metric: str = "sqeuclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Global one-to-one assignment using the Hungarian algorithm.

    If the two populations have different sizes, the method returns
    min(n_source, n_target) matched pairs.
    """
    cost_matrix = compute_cost_matrix(source, target, metric=metric)
    source_idx, target_idx = linear_sum_assignment(cost_matrix)
    return source_idx.astype(int), target_idx.astype(int)


def match_mutual_nearest_neighbors(
    source: ArrayLike,
    target: ArrayLike,
    metric: str = "sqeuclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute one-nearest-neighbor mutual matches.

    Only source-target pairs that are nearest neighbors in both directions are
    retained.
    """
    cost_matrix = compute_cost_matrix(source, target, metric=metric)
    source_to_target = np.argmin(cost_matrix, axis=1)
    target_to_source = np.argmin(cost_matrix, axis=0)

    pairs: List[Tuple[int, int]] = []
    for source_idx, target_idx in enumerate(source_to_target):
        if target_to_source[target_idx] == source_idx:
            pairs.append((source_idx, int(target_idx)))

    if not pairs:
        return np.array([], dtype=int), np.array([], dtype=int)

    pair_array = np.asarray(pairs, dtype=int)
    return pair_array[:, 0], pair_array[:, 1]


def compute_hard_match_perturbations(
    source: ArrayLike,
    target: ArrayLike,
    source_indices: Sequence[int],
    target_indices: Sequence[int],
) -> np.ndarray:
    """
    Compute target-minus-source vectors for a hard matching result.
    """
    source_arr, target_arr = validate_spectral_matrices(source, target)
    source_idx = np.asarray(source_indices, dtype=int)
    target_idx = np.asarray(target_indices, dtype=int)

    if source_idx.shape != target_idx.shape:
        raise ValueError("source_indices and target_indices must have equal shape.")

    if source_idx.ndim != 1:
        raise ValueError("source_indices and target_indices must be one-dimensional.")

    if len(source_idx) == 0:
        return np.empty((0, source_arr.shape[1]), dtype=np.float64)

    return target_arr[target_idx] - source_arr[source_idx]


def iterate_pairwise_conditions(
    data: Mapping[str, np.ndarray],
    source_condition: str,
) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Iterate over one source condition and all other target conditions.

    Parameters
    ----------
    data
        Mapping from condition name to spectral matrix.
    source_condition
        Condition used as the shared source population.

    Yields
    ------
    target_condition, source_matrix, target_matrix
    """
    if source_condition not in data:
        raise KeyError(f"source condition '{source_condition}' was not found.")

    source = np.asarray(data[source_condition], dtype=np.float64)

    for condition, target in data.items():
        if condition == source_condition:
            continue
        yield condition, source, np.asarray(target, dtype=np.float64)


def iterate_trajectory(
    data: Mapping[str, np.ndarray],
    path: Sequence[str],
) -> Iterator[Tuple[int, str, str, np.ndarray, np.ndarray]]:
    """
    Iterate over adjacent population pairs along an ordered trajectory.

    Parameters
    ----------
    data
        Mapping from condition name to spectral matrix.
    path
        Ordered condition names, for example:
        ["dose_0", "dose_1", "dose_2"].

    Yields
    ------
    step, source_name, target_name, source_matrix, target_matrix
    """
    if len(path) < 2:
        raise ValueError("path must contain at least two conditions.")

    missing = [condition for condition in path if condition not in data]
    if missing:
        raise KeyError(f"trajectory conditions not found: {missing}")

    for step, (source_name, target_name) in enumerate(zip(path[:-1], path[1:])):
        yield (
            step,
            source_name,
            target_name,
            np.asarray(data[source_name], dtype=np.float64),
            np.asarray(data[target_name], dtype=np.float64),
        )


__all__ = [
    "TransportResult",
    "PerturbationResult",
    "validate_spectral_matrices",
    "compute_cost_matrix",
    "row_normalize_coupling",
    "compute_emd_coupling",
    "compute_sinkhorn_coupling",
    "compute_barycentric_target",
    "compute_perturbation_vectors",
    "compute_edge_differences",
    "match_nearest_neighbor",
    "match_hungarian",
    "match_mutual_nearest_neighbors",
    "compute_hard_match_perturbations",
    "iterate_pairwise_conditions",
    "iterate_trajectory",
]
