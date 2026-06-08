"""
Mechanism-of-action benchmark for VIP-OT.

This script compares perturbation representations inferred by:

1. Exact balanced optimal transport (OT-EMD)
2. Entropically regularized Sinkhorn OT
3. Greedy nearest-neighbor matching
4. Hungarian assignment
5. Mutual nearest neighbors

For each method and treatment condition, the script computes cell-level
perturbation vectors and a mean difference feature spectrum (DFSP). It then
evaluates mechanism-of-action (MoA) organization using:

- drug-by-drug Pearson correlation matrices
- mean within-MoA correlation
- mean between-MoA correlation
- within-minus-between separation score
- source-cell coverage
- bootstrap robustness

Input spectral table
--------------------
The spectral CSV must contain:

    cell_id, condition, feature_1, feature_2, ...

One condition is designated as the shared control population. All remaining
selected conditions are treated as perturbations.

Input MoA table
---------------
The MoA CSV must contain:

    condition, moa

Example
-------
python code/moa_benchmark.py \
    --spectra data/drug_panel_demo.csv \
    --moa data/moa_metadata.csv \
    --control-condition control \
    --output results/moa_benchmark \
    --n-bootstrap 100
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
    compute_perturbation_vectors,
    compute_sinkhorn_coupling,
    match_hungarian,
    match_mutual_nearest_neighbors,
    match_nearest_neighbor,
)


METHOD_ORDER = [
    "OT-EMD",
    "Sinkhorn",
    "Greedy-NN",
    "Hungarian",
    "MNN",
]


def load_spectral_table(
    path: str | Path,
    cell_id_column: str = "cell_id",
    condition_column: str = "condition",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load a preprocessed single-cell spectral table.

    Returns
    -------
    table
        Input data.
    feature_columns
        Spectral feature columns in their original order.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Spectral file was not found: {path}")

    table = pd.read_csv(path)

    required = {cell_id_column, condition_column}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")

    feature_columns = [
        column
        for column in table.columns
        if column not in {cell_id_column, condition_column}
    ]

    if not feature_columns:
        raise ValueError("No spectral feature columns were found.")

    spectra = table[feature_columns].to_numpy(dtype=np.float64)

    if not np.isfinite(spectra).all():
        raise ValueError("The spectral table contains NaN or Inf values.")

    if table[cell_id_column].duplicated().any():
        raise ValueError(
            f"The spectral table contains duplicate '{cell_id_column}' values."
        )

    table = table.copy()
    table[condition_column] = table[condition_column].astype(str)

    return table, feature_columns


def load_moa_table(
    path: str | Path,
    condition_column: str = "condition",
    moa_column: str = "moa",
) -> pd.DataFrame:
    """Load and validate condition-to-MoA annotations."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MoA metadata file was not found: {path}")

    moa_table = pd.read_csv(path)

    required = {condition_column, moa_column}
    missing = required.difference(moa_table.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")

    moa_table = moa_table[[condition_column, moa_column]].copy()
    moa_table[condition_column] = moa_table[condition_column].astype(str)
    moa_table[moa_column] = moa_table[moa_column].astype(str)

    duplicated = moa_table[condition_column].duplicated(keep=False)
    if duplicated.any():
        duplicate_conditions = sorted(
            moa_table.loc[duplicated, condition_column].unique()
        )
        raise ValueError(
            "Each condition must have one MoA annotation. Duplicates: "
            f"{duplicate_conditions}"
        )

    return moa_table


def build_condition_matrices(
    table: pd.DataFrame,
    feature_columns: Sequence[str],
    condition_column: str,
) -> Dict[str, np.ndarray]:
    """Convert a spectral table into condition-specific matrices."""
    matrices: Dict[str, np.ndarray] = {}

    for condition, group in table.groupby(condition_column, sort=False):
        matrices[str(condition)] = group[list(feature_columns)].to_numpy(
            dtype=np.float64
        )

    return matrices


def infer_method_delta(
    source: np.ndarray,
    target: np.ndarray,
    method: str,
    metric: str = "sqeuclidean",
    sinkhorn_reg: float = 0.05,
) -> Tuple[np.ndarray, float]:
    """
    Infer cell-level perturbation vectors for one matching method.

    Returns
    -------
    delta
        Cell-level target-minus-source perturbation vectors.
    coverage
        Fraction of source cells represented by the method.
    """
    if method == "OT-EMD":
        transport = compute_emd_coupling(source, target, metric=metric)
        result = compute_perturbation_vectors(
            source,
            target,
            transport.row_coupling,
        )
        return result.perturbation_vectors, 1.0

    if method == "Sinkhorn":
        transport = compute_sinkhorn_coupling(
            source,
            target,
            reg=sinkhorn_reg,
            metric=metric,
        )
        result = compute_perturbation_vectors(
            source,
            target,
            transport.row_coupling,
        )
        return result.perturbation_vectors, 1.0

    if method == "Greedy-NN":
        source_idx, target_idx = match_nearest_neighbor(
            source,
            target,
            metric=metric,
        )

    elif method == "Hungarian":
        source_idx, target_idx = match_hungarian(
            source,
            target,
            metric=metric,
        )

    elif method == "MNN":
        source_idx, target_idx = match_mutual_nearest_neighbors(
            source,
            target,
            metric=metric,
        )

    else:
        raise ValueError(f"Unsupported method: {method}")

    delta = compute_hard_match_perturbations(
        source,
        target,
        source_idx,
        target_idx,
    )
    coverage = len(np.unique(source_idx)) / source.shape[0]

    return delta, float(coverage)


def compute_drug_correlation_matrix(
    mean_dfsp: Mapping[str, np.ndarray],
    condition_order: Sequence[str],
) -> pd.DataFrame:
    """Compute Pearson correlation among condition-level mean DFSPs."""
    matrix = np.vstack([mean_dfsp[condition] for condition in condition_order])

    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(matrix)

    return pd.DataFrame(
        correlation,
        index=condition_order,
        columns=condition_order,
    )


def compute_within_between_scores(
    correlation: pd.DataFrame,
    moa_map: Mapping[str, str],
) -> Dict[str, float]:
    """
    Compute within-MoA and between-MoA correlation summaries.
    """
    conditions = list(correlation.index)
    within_values: List[float] = []
    between_values: List[float] = []

    for idx_a in range(len(conditions)):
        for idx_b in range(idx_a + 1, len(conditions)):
            condition_a = conditions[idx_a]
            condition_b = conditions[idx_b]
            value = float(correlation.loc[condition_a, condition_b])

            if not np.isfinite(value):
                continue

            if moa_map[condition_a] == moa_map[condition_b]:
                within_values.append(value)
            else:
                between_values.append(value)

    if not within_values:
        raise ValueError(
            "No within-MoA condition pairs were available. At least one MoA "
            "must contain two or more conditions."
        )

    if not between_values:
        raise ValueError("No between-MoA condition pairs were available.")

    within = np.asarray(within_values, dtype=np.float64)
    between = np.asarray(between_values, dtype=np.float64)

    return {
        "within_mean_corr": float(within.mean()),
        "between_mean_corr": float(between.mean()),
        "separation_score": float(within.mean() - between.mean()),
        "within_std_corr": float(within.std(ddof=1))
        if len(within) > 1
        else np.nan,
        "between_std_corr": float(between.std(ddof=1))
        if len(between) > 1
        else np.nan,
        "n_within_pairs": int(len(within)),
        "n_between_pairs": int(len(between)),
    }


def run_full_panel(
    condition_matrices: Mapping[str, np.ndarray],
    control_condition: str,
    treatment_conditions: Sequence[str],
    moa_map: Mapping[str, str],
    metric: str = "sqeuclidean",
    sinkhorn_reg: float = 0.05,
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, pd.DataFrame],
    pd.DataFrame,
]:
    """
    Run all matching methods on the complete treatment panel.

    Returns
    -------
    all_deltas
        method -> condition -> cell-level perturbation vectors.
    correlation_matrices
        method -> condition-level DFSP correlation matrix.
    score_table
        MoA separation and coverage summaries.
    """
    source = condition_matrices[control_condition]

    all_deltas: Dict[str, Dict[str, np.ndarray]] = {
        method: {} for method in METHOD_ORDER
    }
    coverage_rows: List[Dict[str, float]] = []

    for method in METHOD_ORDER:
        for condition in treatment_conditions:
            target = condition_matrices[condition]

            delta, coverage = infer_method_delta(
                source,
                target,
                method=method,
                metric=metric,
                sinkhorn_reg=sinkhorn_reg,
            )

            if delta.shape[0] == 0:
                raise ValueError(
                    f"{method} returned no matched cells for condition '{condition}'."
                )

            all_deltas[method][condition] = delta

            coverage_rows.append(
                {
                    "method": method,
                    "condition": condition,
                    "n_perturbation_vectors": int(delta.shape[0]),
                    "coverage": coverage,
                }
            )

    coverage_table = pd.DataFrame(coverage_rows)
    correlation_matrices: Dict[str, pd.DataFrame] = {}
    score_rows: List[Dict[str, float]] = []

    for method in METHOD_ORDER:
        mean_dfsp = {
            condition: all_deltas[method][condition].mean(axis=0)
            for condition in treatment_conditions
        }

        correlation = compute_drug_correlation_matrix(
            mean_dfsp,
            treatment_conditions,
        )
        correlation_matrices[method] = correlation

        score = compute_within_between_scores(correlation, moa_map)
        score_rows.append(
            {
                "method": method,
                **score,
                "mean_coverage": float(
                    coverage_table.loc[
                        coverage_table["method"] == method,
                        "coverage",
                    ].mean()
                ),
            }
        )

    score_table = pd.DataFrame(score_rows).sort_values(
        "separation_score",
        ascending=False,
    )

    return all_deltas, correlation_matrices, score_table, coverage_table


def bootstrap_panel(
    condition_matrices: Mapping[str, np.ndarray],
    control_condition: str,
    treatment_conditions: Sequence[str],
    moa_map: Mapping[str, str],
    n_bootstrap: int = 100,
    metric: str = "sqeuclidean",
    sinkhorn_reg: float = 0.05,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Bootstrap source and target cells with replacement and recompute scores.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")

    source_full = condition_matrices[control_condition]
    rng_master = np.random.default_rng(random_state)
    bootstrap_rows: List[Dict[str, float]] = []

    for bootstrap_idx in range(n_bootstrap):
        rng = np.random.default_rng(
            rng_master.integers(0, np.iinfo(np.int32).max)
        )

        source_indices = rng.integers(
            0,
            source_full.shape[0],
            size=source_full.shape[0],
        )
        source_bootstrap = source_full[source_indices]

        mean_dfsp_by_method: Dict[str, Dict[str, np.ndarray]] = {
            method: {} for method in METHOD_ORDER
        }
        coverage_by_method: Dict[str, List[float]] = {
            method: [] for method in METHOD_ORDER
        }

        for condition in treatment_conditions:
            target_full = condition_matrices[condition]
            target_indices = rng.integers(
                0,
                target_full.shape[0],
                size=target_full.shape[0],
            )
            target_bootstrap = target_full[target_indices]

            for method in METHOD_ORDER:
                delta, coverage = infer_method_delta(
                    source_bootstrap,
                    target_bootstrap,
                    method=method,
                    metric=metric,
                    sinkhorn_reg=sinkhorn_reg,
                )

                coverage_by_method[method].append(coverage)

                if delta.shape[0] == 0:
                    mean_dfsp_by_method[method][condition] = np.full(
                        source_full.shape[1],
                        np.nan,
                    )
                else:
                    mean_dfsp_by_method[method][condition] = delta.mean(axis=0)

        for method in METHOD_ORDER:
            valid_conditions = [
                condition
                for condition in treatment_conditions
                if np.isfinite(
                    mean_dfsp_by_method[method][condition]
                ).all()
            ]

            if len(valid_conditions) < 2:
                continue

            valid_moa_map = {
                condition: moa_map[condition]
                for condition in valid_conditions
            }

            try:
                correlation = compute_drug_correlation_matrix(
                    mean_dfsp_by_method[method],
                    valid_conditions,
                )
                scores = compute_within_between_scores(
                    correlation,
                    valid_moa_map,
                )
            except ValueError:
                continue

            bootstrap_rows.append(
                {
                    "bootstrap": bootstrap_idx,
                    "method": method,
                    **scores,
                    "mean_coverage": float(
                        np.mean(coverage_by_method[method])
                    ),
                    "n_valid_conditions": int(len(valid_conditions)),
                }
            )

    return pd.DataFrame(bootstrap_rows)


def save_cell_level_deltas(
    all_deltas: Mapping[str, Mapping[str, np.ndarray]],
    output_dir: Path,
) -> None:
    """Save cell-level perturbation matrices as compressed NPZ files."""
    delta_dir = output_dir / "cell_level_deltas"
    delta_dir.mkdir(parents=True, exist_ok=True)

    for method, condition_dict in all_deltas.items():
        safe_method = method.lower().replace("-", "_")
        np.savez_compressed(
            delta_dir / f"{safe_method}.npz",
            **condition_dict,
        )


def run_benchmark(
    spectra_path: str | Path,
    moa_path: str | Path,
    control_condition: str,
    output_dir: str | Path,
    cell_id_column: str = "cell_id",
    condition_column: str = "condition",
    moa_column: str = "moa",
    include_conditions: Sequence[str] | None = None,
    metric: str = "sqeuclidean",
    sinkhorn_reg: float = 0.05,
    n_bootstrap: int = 100,
    random_state: int = 0,
    save_cell_deltas: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the complete MoA benchmark and save outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spectral_table, feature_columns = load_spectral_table(
        spectra_path,
        cell_id_column=cell_id_column,
        condition_column=condition_column,
    )
    moa_table = load_moa_table(
        moa_path,
        condition_column=condition_column,
        moa_column=moa_column,
    )

    condition_matrices = build_condition_matrices(
        spectral_table,
        feature_columns,
        condition_column,
    )

    if control_condition not in condition_matrices:
        raise ValueError(
            f"Control condition '{control_condition}' was not found."
        )

    moa_map_all = dict(
        zip(
            moa_table[condition_column],
            moa_table[moa_column],
        )
    )

    available_treatments = [
        condition
        for condition in condition_matrices
        if condition != control_condition
    ]

    if include_conditions:
        treatment_conditions = [str(value) for value in include_conditions]
        missing = [
            condition
            for condition in treatment_conditions
            if condition not in condition_matrices
        ]
        if missing:
            raise ValueError(
                f"Selected treatment conditions were not found: {missing}"
            )
    else:
        treatment_conditions = available_treatments

    missing_moa = [
        condition
        for condition in treatment_conditions
        if condition not in moa_map_all
    ]
    if missing_moa:
        raise ValueError(
            f"Missing MoA annotations for condition(s): {missing_moa}"
        )

    moa_map = {
        condition: moa_map_all[condition]
        for condition in treatment_conditions
    }

    (
        all_deltas,
        correlation_matrices,
        score_table,
        coverage_table,
    ) = run_full_panel(
        condition_matrices=condition_matrices,
        control_condition=control_condition,
        treatment_conditions=treatment_conditions,
        moa_map=moa_map,
        metric=metric,
        sinkhorn_reg=sinkhorn_reg,
    )

    coverage_table.to_csv(
        output_dir / "coverage_summary.csv",
        index=False,
    )
    score_table.to_csv(
        output_dir / "moa_separation_scores.csv",
        index=False,
    )

    correlation_dir = output_dir / "correlation_matrices"
    correlation_dir.mkdir(parents=True, exist_ok=True)

    for method, matrix in correlation_matrices.items():
        safe_method = method.lower().replace("-", "_")
        matrix.to_csv(
            correlation_dir / f"{safe_method}_drug_correlation.csv"
        )

    bootstrap_table = bootstrap_panel(
        condition_matrices=condition_matrices,
        control_condition=control_condition,
        treatment_conditions=treatment_conditions,
        moa_map=moa_map,
        n_bootstrap=n_bootstrap,
        metric=metric,
        sinkhorn_reg=sinkhorn_reg,
        random_state=random_state,
    )
    bootstrap_table.to_csv(
        output_dir / "bootstrap_scores.csv",
        index=False,
    )

    if save_cell_deltas:
        save_cell_level_deltas(all_deltas, output_dir)

    metadata = {
        "spectral_file": str(Path(spectra_path)),
        "moa_file": str(Path(moa_path)),
        "control_condition": control_condition,
        "treatment_conditions": treatment_conditions,
        "n_treatment_conditions": len(treatment_conditions),
        "n_features": len(feature_columns),
        "distance_metric": metric,
        "sinkhorn_regularization": sinkhorn_reg,
        "n_bootstrap": n_bootstrap,
        "random_state": random_state,
        "methods": METHOD_ORDER,
    }

    with open(
        output_dir / "run_metadata.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, indent=2)

    return score_table, bootstrap_table


def parse_condition_list(value: str | None) -> List[str] | None:
    """Parse a comma-separated condition list."""
    if value is None:
        return None

    conditions = [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]
    return conditions or None


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run the VIP-OT MoA matching-method benchmark."
    )

    parser.add_argument(
        "--spectra",
        required=True,
        help="CSV containing preprocessed single-cell spectra.",
    )
    parser.add_argument(
        "--moa",
        required=True,
        help="CSV mapping treatment conditions to MoA labels.",
    )
    parser.add_argument(
        "--control-condition",
        required=True,
        help="Name of the shared control condition.",
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
        "--condition-column",
        default="condition",
        help="Condition column. Default: condition",
    )
    parser.add_argument(
        "--moa-column",
        default="moa",
        help="MoA annotation column. Default: moa",
    )
    parser.add_argument(
        "--include-conditions",
        default=None,
        help="Optional comma-separated treatment-condition subset.",
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
        "--n-bootstrap",
        type=int,
        default=100,
        help="Number of bootstrap repetitions. Default: 100",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed. Default: 0",
    )
    parser.add_argument(
        "--save-cell-deltas",
        action="store_true",
        help="Save cell-level perturbation matrices as compressed NPZ files.",
    )

    return parser


def main() -> None:
    """Command-line entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    score_table, bootstrap_table = run_benchmark(
        spectra_path=args.spectra,
        moa_path=args.moa,
        control_condition=args.control_condition,
        output_dir=args.output,
        cell_id_column=args.cell_id_column,
        condition_column=args.condition_column,
        moa_column=args.moa_column,
        include_conditions=parse_condition_list(args.include_conditions),
        metric=args.metric,
        sinkhorn_reg=args.sinkhorn_reg,
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
        save_cell_deltas=args.save_cell_deltas,
    )

    print("\nMoA benchmark completed.\n")
    print(score_table.to_string(index=False))
    print(
        f"\nBootstrap rows saved: {len(bootstrap_table)}"
    )
    print(f"Results written to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
