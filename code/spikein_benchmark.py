"""
Spike-in identity-preservation benchmark for VIP-OT.

This script compares five matching strategies on a mixed-population
control-versus-treated benchmark:

1. Greedy nearest-neighbor matching
2. Hungarian assignment
3. Mutual nearest neighbors
4. Exact balanced optimal transport (EMD)
5. Entropically regularized Sinkhorn OT

The script evaluates:

- source-cell coverage
- same-line identity preservation
- soft within-line transport mass
- hard argmax purity for soft couplings
- source-line to target-line flow matrices
- label-permutation null distributions

Input format
------------
Source and target files must be CSV tables containing:

    cell_id, cell_line, feature_1, feature_2, ...

The metadata-column names can be changed through command-line arguments.
All remaining columns are treated as preprocessed spectral features.

Example
-------
python code/spikein_benchmark.py \
    --source data/spikein_source.csv \
    --target data/spikein_target.csv \
    --output results/spikein
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from vipot import (
    compute_emd_coupling,
    compute_hard_match_perturbations,
    compute_sinkhorn_coupling,
    match_hungarian,
    match_mutual_nearest_neighbors,
    match_nearest_neighbor,
)


def load_labeled_spectra(
    path: str | Path,
    cell_id_column: str = "cell_id",
    label_column: str = "cell_line",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Load one labeled single-cell spectral table.

    Parameters
    ----------
    path
        CSV file containing metadata and spectral columns.
    cell_id_column
        Column containing unique cell identifiers.
    label_column
        Column containing cell-line or identity labels.

    Returns
    -------
    table
        Original input table.
    spectra
        Spectral matrix with shape (n_cells, n_features).
    labels
        Identity labels.
    feature_columns
        Names of spectral columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file was not found: {path}")

    table = pd.read_csv(path)

    required = {cell_id_column, label_column}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")

    feature_columns = [
        column
        for column in table.columns
        if column not in {cell_id_column, label_column}
    ]

    if not feature_columns:
        raise ValueError(f"No spectral feature columns were found in {path}.")

    spectra = table[feature_columns].to_numpy(dtype=np.float64)
    labels = table[label_column].astype(str).to_numpy()

    if not np.isfinite(spectra).all():
        raise ValueError(f"{path} contains NaN or Inf spectral values.")

    if table[cell_id_column].duplicated().any():
        raise ValueError(f"{path} contains duplicated values in '{cell_id_column}'.")

    return table, spectra, labels, feature_columns


def validate_source_target_features(
    source_features: Sequence[str],
    target_features: Sequence[str],
) -> None:
    """Ensure that source and target spectral columns are identical and ordered."""
    if list(source_features) != list(target_features):
        raise ValueError(
            "Source and target spectral columns must be identical and in the same order."
        )


def evaluate_hard_matching(
    method_name: str,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    Evaluate identity preservation for a hard matching method.

    Returns
    -------
    metrics
        Summary metrics.
    flow_matrix
        Source-label-normalized confusion matrix.
    pair_table
        One row per matched source-target pair.
    """
    source_indices = np.asarray(source_indices, dtype=int)
    target_indices = np.asarray(target_indices, dtype=int)

    if source_indices.shape != target_indices.shape:
        raise ValueError("source_indices and target_indices must have equal shape.")

    n_source = len(source_labels)
    matched_sources = np.unique(source_indices)
    coverage = len(matched_sources) / n_source if n_source else np.nan

    if len(source_indices) == 0:
        source_order = list(pd.unique(source_labels))
        target_order = list(pd.unique(target_labels))
        flow_matrix = pd.DataFrame(
            0.0,
            index=source_order,
            columns=target_order,
        )
        pair_table = pd.DataFrame(
            columns=[
                "source_idx",
                "target_idx",
                "source_line",
                "target_line",
                "same_line",
            ]
        )
        metrics = {
            "method": method_name,
            "matching_type": "hard",
            "coverage": coverage,
            "n_pairs": 0,
            "identity_score": np.nan,
            "same_line_fraction": np.nan,
            "cross_line_fraction": np.nan,
            "hard_argmax_purity": np.nan,
            "mean_within_line_mass_fraction": np.nan,
        }
        return metrics, flow_matrix, pair_table

    pair_table = pd.DataFrame(
        {
            "source_idx": source_indices,
            "target_idx": target_indices,
            "source_line": source_labels[source_indices],
            "target_line": target_labels[target_indices],
        }
    )
    pair_table["same_line"] = (
        pair_table["source_line"] == pair_table["target_line"]
    ).astype(int)

    same_line_fraction = float(pair_table["same_line"].mean())

    flow_matrix = pd.crosstab(
        pair_table["source_line"],
        pair_table["target_line"],
        normalize="index",
    )

    source_order = list(pd.unique(source_labels))
    target_order = list(pd.unique(target_labels))
    flow_matrix = flow_matrix.reindex(
        index=source_order,
        columns=target_order,
        fill_value=0.0,
    )

    metrics = {
        "method": method_name,
        "matching_type": "hard",
        "coverage": coverage,
        "n_pairs": int(len(pair_table)),
        "identity_score": same_line_fraction,
        "same_line_fraction": same_line_fraction,
        "cross_line_fraction": 1.0 - same_line_fraction,
        "hard_argmax_purity": same_line_fraction,
        "mean_within_line_mass_fraction": np.nan,
    }

    return metrics, flow_matrix, pair_table


def compute_block_mass(
    row_coupling: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
) -> pd.DataFrame:
    """Aggregate a row-normalized coupling into label-to-label transport mass."""
    source_order = list(pd.unique(source_labels))
    target_order = list(pd.unique(target_labels))

    block = pd.DataFrame(
        0.0,
        index=source_order,
        columns=target_order,
    )

    for source_label in source_order:
        source_mask = source_labels == source_label
        for target_label in target_order:
            target_mask = target_labels == target_label
            block.loc[source_label, target_label] = row_coupling[
                source_mask
            ][:, target_mask].sum()

    return block


def per_cell_within_line_mass(
    row_coupling: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
) -> np.ndarray:
    """Compute within-line transport mass for every source cell."""
    values = np.zeros(len(source_labels), dtype=np.float64)

    for source_idx, source_label in enumerate(source_labels):
        same_target = target_labels == source_label
        values[source_idx] = row_coupling[source_idx, same_target].sum()

    return values


def evaluate_soft_coupling(
    method_name: str,
    row_coupling: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate identity preservation using the full soft coupling.

    Returns
    -------
    metrics
        Summary metrics.
    normalized_flow
        Source-label-normalized transport flow.
    per_cell
        Per-source-cell identity scores.
    block_mass
        Unnormalized block transport totals after row normalization.
    """
    row_coupling = np.asarray(row_coupling, dtype=np.float64)
    block_mass = compute_block_mass(row_coupling, source_labels, target_labels)
    normalized_flow = block_mass.div(block_mass.sum(axis=1), axis=0)

    within_line = per_cell_within_line_mass(
        row_coupling,
        source_labels,
        target_labels,
    )

    argmax_target_idx = np.argmax(row_coupling, axis=1)
    argmax_target_line = target_labels[argmax_target_idx]
    argmax_correct = (argmax_target_line == source_labels).astype(int)

    per_cell = pd.DataFrame(
        {
            "source_idx": np.arange(len(source_labels), dtype=int),
            "source_line": source_labels,
            "within_line_mass_fraction": within_line,
            "cross_line_mass_fraction": 1.0 - within_line,
            "argmax_target_idx": argmax_target_idx,
            "argmax_target_line": argmax_target_line,
            "argmax_correct": argmax_correct,
        }
    )

    total_mass = float(block_mass.to_numpy().sum())
    diagonal_mass = 0.0
    for label in pd.unique(source_labels):
        if label in block_mass.index and label in block_mass.columns:
            diagonal_mass += float(block_mass.loc[label, label])

    same_line_mass_fraction = diagonal_mass / total_mass
    hard_argmax_purity = float(argmax_correct.mean())
    mean_within_line = float(within_line.mean())

    metrics = {
        "method": method_name,
        "matching_type": "soft",
        "coverage": 1.0,
        "n_pairs": int(len(source_labels)),
        "identity_score": same_line_mass_fraction,
        "same_line_fraction": np.nan,
        "cross_line_fraction": 1.0 - same_line_mass_fraction,
        "hard_argmax_purity": hard_argmax_purity,
        "mean_within_line_mass_fraction": mean_within_line,
    }

    return metrics, normalized_flow, per_cell, block_mass


def permutation_null(
    row_coupling: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    n_permutations: int = 1000,
    random_state: int = 0,
) -> Tuple[float, np.ndarray, float]:
    """
    Generate a target-label permutation null for soft identity preservation.

    The coupling remains fixed while target labels are shuffled.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1.")

    rng = np.random.default_rng(random_state)

    observed = float(
        per_cell_within_line_mass(
            row_coupling,
            source_labels,
            target_labels,
        ).mean()
    )

    null_values = np.empty(n_permutations, dtype=np.float64)

    for permutation_idx in range(n_permutations):
        shuffled_labels = rng.permutation(target_labels)
        null_values[permutation_idx] = per_cell_within_line_mass(
            row_coupling,
            source_labels,
            shuffled_labels,
        ).mean()

    empirical_p = (
        np.count_nonzero(null_values >= observed) + 1
    ) / (n_permutations + 1)

    return observed, null_values, float(empirical_p)


def save_flow_matrices(
    flow_matrices: Mapping[str, pd.DataFrame],
    output_path: str | Path,
) -> None:
    """Save one flow matrix per worksheet."""
    output_path = Path(output_path)
    with pd.ExcelWriter(output_path) as writer:
        for method, matrix in flow_matrices.items():
            sheet_name = method[:31]
            matrix.to_excel(writer, sheet_name=sheet_name)


def run_benchmark(
    source_path: str | Path,
    target_path: str | Path,
    output_dir: str | Path,
    cell_id_column: str = "cell_id",
    label_column: str = "cell_line",
    metric: str = "sqeuclidean",
    sinkhorn_reg: float = 0.05,
    n_permutations: int = 1000,
    random_state: int = 0,
    save_couplings: bool = False,
) -> pd.DataFrame:
    """Run the complete mixed-population identity benchmark."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_table, source_X, source_labels, source_features = load_labeled_spectra(
        source_path,
        cell_id_column=cell_id_column,
        label_column=label_column,
    )
    target_table, target_X, target_labels, target_features = load_labeled_spectra(
        target_path,
        cell_id_column=cell_id_column,
        label_column=label_column,
    )

    validate_source_target_features(source_features, target_features)

    source_ids = source_table[cell_id_column].astype(str).to_numpy()
    target_ids = target_table[cell_id_column].astype(str).to_numpy()

    summary_rows: List[Dict[str, float]] = []
    flow_matrices: Dict[str, pd.DataFrame] = {}
    per_cell_tables: List[pd.DataFrame] = []
    pair_tables: List[pd.DataFrame] = []
    permutation_rows: List[Dict[str, float]] = []

    hard_methods = {
        "Greedy-NN": match_nearest_neighbor,
        "Hungarian": match_hungarian,
        "MNN": match_mutual_nearest_neighbors,
    }

    for method_name, matching_function in hard_methods.items():
        source_idx, target_idx = matching_function(
            source_X,
            target_X,
            metric=metric,
        )

        metrics, flow_matrix, pair_table = evaluate_hard_matching(
            method_name,
            source_idx,
            target_idx,
            source_labels,
            target_labels,
        )

        pair_table["source_cell_id"] = source_ids[
            pair_table["source_idx"].to_numpy(dtype=int)
        ] if len(pair_table) else pd.Series(dtype=str)

        pair_table["target_cell_id"] = target_ids[
            pair_table["target_idx"].to_numpy(dtype=int)
        ] if len(pair_table) else pd.Series(dtype=str)

        pair_table["method"] = method_name

        summary_rows.append(metrics)
        flow_matrices[method_name] = flow_matrix
        pair_tables.append(pair_table)

    soft_results = {
        "OT-EMD": compute_emd_coupling(
            source_X,
            target_X,
            metric=metric,
        ),
        "Sinkhorn": compute_sinkhorn_coupling(
            source_X,
            target_X,
            reg=sinkhorn_reg,
            metric=metric,
        ),
    }

    for method_name, transport in soft_results.items():
        metrics, flow_matrix, per_cell, block_mass = evaluate_soft_coupling(
            method_name,
            transport.row_coupling,
            source_labels,
            target_labels,
        )

        per_cell["source_cell_id"] = source_ids
        per_cell["method"] = method_name

        observed, null_values, empirical_p = permutation_null(
            transport.row_coupling,
            source_labels,
            target_labels,
            n_permutations=n_permutations,
            random_state=random_state,
        )

        metrics["permutation_p_value"] = empirical_p
        metrics["permutation_null_mean"] = float(null_values.mean())
        metrics["permutation_null_std"] = float(null_values.std(ddof=1))

        permutation_table = pd.DataFrame(
            {
                "method": method_name,
                "permutation": np.arange(n_permutations, dtype=int),
                "null_within_line_mass_fraction": null_values,
                "observed_within_line_mass_fraction": observed,
                "empirical_p_value": empirical_p,
            }
        )
        permutation_rows.append(permutation_table)

        summary_rows.append(metrics)
        flow_matrices[method_name] = flow_matrix
        flow_matrices[f"{method_name}_block_mass"] = block_mass
        per_cell_tables.append(per_cell)

        if save_couplings:
            np.savez_compressed(
                output_dir / f"{method_name.lower().replace('-', '_')}_coupling.npz",
                raw_coupling=transport.raw_coupling,
                row_coupling=transport.row_coupling,
                cost_matrix=transport.cost_matrix,
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "benchmark_summary.csv", index=False)

    if pair_tables:
        pd.concat(pair_tables, ignore_index=True).to_csv(
            output_dir / "hard_matching_pairs.csv",
            index=False,
        )

    if per_cell_tables:
        pd.concat(per_cell_tables, ignore_index=True).to_csv(
            output_dir / "soft_coupling_per_cell.csv",
            index=False,
        )

    if permutation_rows:
        pd.concat(permutation_rows, ignore_index=True).to_csv(
            output_dir / "permutation_null.csv",
            index=False,
        )

    save_flow_matrices(
        flow_matrices,
        output_dir / "identity_flow_matrices.xlsx",
    )

    run_metadata = {
        "source_file": str(Path(source_path)),
        "target_file": str(Path(target_path)),
        "n_source_cells": int(source_X.shape[0]),
        "n_target_cells": int(target_X.shape[0]),
        "n_features": int(source_X.shape[1]),
        "source_labels": pd.Series(source_labels).value_counts().to_dict(),
        "target_labels": pd.Series(target_labels).value_counts().to_dict(),
        "distance_metric": metric,
        "sinkhorn_regularization": sinkhorn_reg,
        "n_permutations": n_permutations,
        "random_state": random_state,
    }

    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2)

    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    """Create command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run the VIP-OT mixed-population spike-in benchmark."
    )

    parser.add_argument(
        "--source",
        required=True,
        help="CSV containing mixed control/source cells.",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="CSV containing mixed treated/target cells.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--cell-id-column",
        default="cell_id",
        help="Cell identifier column. Default: cell_id",
    )
    parser.add_argument(
        "--label-column",
        default="cell_line",
        help="Identity-label column. Default: cell_line",
    )
    parser.add_argument(
        "--metric",
        default="sqeuclidean",
        help="Pairwise distance metric. Default: sqeuclidean",
    )
    parser.add_argument(
        "--sinkhorn-reg",
        type=float,
        default=0.05,
        help="Sinkhorn regularization strength. Default: 0.05",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Number of target-label permutations. Default: 1000",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed. Default: 0",
    )
    parser.add_argument(
        "--save-couplings",
        action="store_true",
        help="Save full OT and Sinkhorn coupling matrices.",
    )

    return parser


def main() -> None:
    """Command-line entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    summary = run_benchmark(
        source_path=args.source,
        target_path=args.target,
        output_dir=args.output,
        cell_id_column=args.cell_id_column,
        label_column=args.label_column,
        metric=args.metric,
        sinkhorn_reg=args.sinkhorn_reg,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
        save_couplings=args.save_couplings,
    )

    print("\nSpike-in benchmark completed.\n")
    print(summary.to_string(index=False))
    print(f"\nResults written to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
