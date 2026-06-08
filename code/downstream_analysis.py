"""
Downstream VIP-OT analyses.

This script combines three representative downstream applications:

1. Baseline-dependent response heterogeneity
2. Dose-indexed Spectral Velocity
3. Drug-combination path analysis

The analyses begin from preprocessed single-cell spectral matrices. Raw-image
reconstruction, segmentation, instrument-specific quality control, and other
platform-dependent preprocessing steps are not included.

Input spectral tables
---------------------
Each CSV should contain:

    cell_id, condition, feature_1, feature_2, ...

Spectral column names may be numeric wavenumbers, such as:

    1000.0, 1003.8, 1007.7, ...

or arbitrary feature names. Analyses that use spectral regions require numeric
spectral-column names.

Examples
--------
Heterogeneity:

python code/downstream_analysis.py heterogeneity \
    --spectra data/heterogeneity_demo.csv \
    --source-condition control \
    --target-condition anisomycin \
    --metric-numerator 2070,2150 \
    --metric-denominator 2800,3000 \
    --baseline-region 1720,1760 \
    --output results/heterogeneity

Spectral Velocity:

python code/downstream_analysis.py velocity \
    --spectra data/dose_series_demo.csv \
    --path control,dose_1,dose_2,dose_3 \
    --output results/velocity

Combination paths:

python code/downstream_analysis.py combination \
    --spectra data/combination_demo.csv \
    --path gefitinib,gefitinib_bortezomib \
    --path bortezomib,gefitinib_bortezomib \
    --output results/combination
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, silhouette_score

from vipot import (
    compute_emd_coupling,
    compute_perturbation_vectors,
)


# =========================================================
# Shared utilities
# =========================================================


def load_spectral_table(
    path: str | Path,
    cell_id_column: str = "cell_id",
    condition_column: str = "condition",
) -> Tuple[pd.DataFrame, List[str]]:
    """Load a preprocessed single-cell spectral table."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file was not found: {path}")

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

    values = table[feature_columns].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError("Spectral data contain NaN or Inf values.")

    if table[cell_id_column].duplicated().any():
        raise ValueError(
            f"The input contains duplicated '{cell_id_column}' values."
        )

    table = table.copy()
    table[cell_id_column] = table[cell_id_column].astype(str)
    table[condition_column] = table[condition_column].astype(str)

    return table, feature_columns


def build_condition_matrices(
    table: pd.DataFrame,
    feature_columns: Sequence[str],
    condition_column: str,
) -> Dict[str, np.ndarray]:
    """Create one spectral matrix per condition."""
    return {
        str(condition): group[list(feature_columns)].to_numpy(dtype=np.float64)
        for condition, group in table.groupby(condition_column, sort=False)
    }


def build_condition_ids(
    table: pd.DataFrame,
    cell_id_column: str,
    condition_column: str,
) -> Dict[str, np.ndarray]:
    """Create one cell-ID array per condition."""
    return {
        str(condition): group[cell_id_column].astype(str).to_numpy()
        for condition, group in table.groupby(condition_column, sort=False)
    }


def parse_numeric_wavenumbers(
    feature_columns: Sequence[str],
) -> np.ndarray:
    """Convert spectral column names to numeric wavenumbers."""
    try:
        return np.asarray([float(column) for column in feature_columns])
    except ValueError as exc:
        raise ValueError(
            "This analysis requires numeric spectral-column names."
        ) from exc


def parse_interval(value: str) -> Tuple[float, float]:
    """Parse an interval formatted as 'start,stop'."""
    parts = [item.strip() for item in value.split(",")]

    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "Intervals must be formatted as 'start,stop'."
        )

    start, stop = map(float, parts)

    if start >= stop:
        raise argparse.ArgumentTypeError(
            "The interval start must be smaller than the stop."
        )

    return start, stop


def spectral_region_mask(
    wavenumbers: np.ndarray,
    interval: Tuple[float, float],
) -> np.ndarray:
    """Return a Boolean mask for an inclusive spectral interval."""
    start, stop = interval
    mask = (wavenumbers >= start) & (wavenumbers <= stop)

    if not mask.any():
        raise ValueError(
            f"No spectral features fall within interval {interval}."
        )

    return mask


def average_spectral_region(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    interval: Tuple[float, float],
) -> np.ndarray:
    """Average each cell spectrum within a selected interval."""
    mask = spectral_region_mask(wavenumbers, interval)
    return spectra[:, mask].mean(axis=1)


def compute_ratio_metric(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    numerator_interval: Tuple[float, float],
    denominator_interval: Tuple[float, float],
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute a per-cell ratio between two mean spectral regions."""
    numerator = average_spectral_region(
        spectra,
        wavenumbers,
        numerator_interval,
    )
    denominator = average_spectral_region(
        spectra,
        wavenumbers,
        denominator_interval,
    )

    denominator = np.where(np.abs(denominator) < eps, eps, denominator)
    return numerator / denominator


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cliff's delta.

    Positive values indicate that values in x tend to exceed values in y.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) == 0 or len(y) == 0:
        return np.nan

    differences = x[:, None] - y[None, :]
    greater = np.count_nonzero(differences > 0)
    smaller = np.count_nonzero(differences < 0)

    return float((greater - smaller) / differences.size)


def save_json(data: Mapping, path: str | Path) -> None:
    """Save a dictionary as formatted JSON."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


# =========================================================
# Heterogeneity analysis
# =========================================================


def cluster_treated_metric(
    metric_values: np.ndarray,
    n_clusters: int = 2,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Cluster treated-cell metric values and order labels by cluster center.

    For two clusters, returned integer labels are:
        0 = low
        1 = high
    """
    metric_values = np.asarray(metric_values, dtype=np.float64)

    if metric_values.ndim != 1:
        raise ValueError("metric_values must be one-dimensional.")

    if len(metric_values) < n_clusters:
        raise ValueError("Fewer treated cells than requested clusters.")

    model = KMeans(
        n_clusters=n_clusters,
        n_init=20,
        random_state=random_state,
    )
    raw_labels = model.fit_predict(metric_values[:, None])
    raw_centers = model.cluster_centers_.ravel()

    center_order = np.argsort(raw_centers)
    remap = {
        raw_label: ordered_label
        for ordered_label, raw_label in enumerate(center_order)
    }

    ordered_labels = np.asarray(
        [remap[label] for label in raw_labels],
        dtype=int,
    )
    ordered_centers = raw_centers[center_order]

    score = (
        float(silhouette_score(metric_values[:, None], ordered_labels))
        if len(np.unique(ordered_labels)) > 1
        else np.nan
    )

    return ordered_labels, ordered_centers, score


def propagate_target_labels(
    row_coupling: np.ndarray,
    target_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate treated-cell labels to source cells through OT weights.

    The returned hard source label is the class receiving the largest
    cumulative transport mass. The probability matrix preserves the full soft
    label assignment.
    """
    target_labels = np.asarray(target_labels)
    classes = np.sort(np.unique(target_labels))

    probabilities = np.column_stack(
        [
            row_coupling[:, target_labels == class_label].sum(axis=1)
            for class_label in classes
        ]
    )

    hard_labels = classes[np.argmax(probabilities, axis=1)]
    return hard_labels, probabilities


def run_heterogeneity_analysis(
    spectra_path: str | Path,
    source_condition: str,
    target_condition: str,
    metric_numerator: Tuple[float, float],
    metric_denominator: Tuple[float, float],
    baseline_region: Tuple[float, float],
    output_dir: str | Path,
    cell_id_column: str = "cell_id",
    condition_column: str = "condition",
    random_state: int = 0,
) -> None:
    """Run baseline-dependent response heterogeneity analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table, feature_columns = load_spectral_table(
        spectra_path,
        cell_id_column=cell_id_column,
        condition_column=condition_column,
    )
    wavenumbers = parse_numeric_wavenumbers(feature_columns)

    matrices = build_condition_matrices(
        table,
        feature_columns,
        condition_column,
    )
    ids = build_condition_ids(
        table,
        cell_id_column,
        condition_column,
    )

    for condition in [source_condition, target_condition]:
        if condition not in matrices:
            raise ValueError(f"Condition '{condition}' was not found.")

    source = matrices[source_condition]
    target = matrices[target_condition]

    transport = compute_emd_coupling(source, target)
    perturbation = compute_perturbation_vectors(
        source,
        target,
        transport.row_coupling,
    )

    target_metric = compute_ratio_metric(
        target,
        wavenumbers,
        metric_numerator,
        metric_denominator,
    )
    source_metric = compute_ratio_metric(
        source,
        wavenumbers,
        metric_numerator,
        metric_denominator,
    )
    barycentric_metric = compute_ratio_metric(
        perturbation.barycentric_target,
        wavenumbers,
        metric_numerator,
        metric_denominator,
    )

    target_labels, cluster_centers, silhouette = cluster_treated_metric(
        target_metric,
        n_clusters=2,
        random_state=random_state,
    )

    source_labels, source_probabilities = propagate_target_labels(
        transport.row_coupling,
        target_labels,
    )

    baseline_values = average_spectral_region(
        source,
        wavenumbers,
        baseline_region,
    )

    low_values = baseline_values[source_labels == 0]
    high_values = baseline_values[source_labels == 1]

    mann_whitney = mannwhitneyu(
        low_values,
        high_values,
        alternative="two-sided",
    )
    delta = cliffs_delta(low_values, high_values)

    if len(np.unique(source_labels)) == 2:
        auc_raw = roc_auc_score(source_labels, baseline_values)
        auc = float(max(auc_raw, 1.0 - auc_raw))
    else:
        auc = np.nan

    target_table = pd.DataFrame(
        {
            "cell_id": ids[target_condition],
            "treated_metric": target_metric,
            "treated_cluster": target_labels,
            "treated_cluster_name": np.where(
                target_labels == 0,
                "low",
                "high",
            ),
        }
    )
    target_table.to_csv(
        output_dir / "treated_state_labels.csv",
        index=False,
    )

    source_table = pd.DataFrame(
        {
            "cell_id": ids[source_condition],
            "baseline_metric": source_metric,
            "barycentric_target_metric": barycentric_metric,
            "response_delta": barycentric_metric - source_metric,
            "baseline_region_value": baseline_values,
            "propagated_cluster": source_labels,
            "propagated_cluster_name": np.where(
                source_labels == 0,
                "low",
                "high",
            ),
            "probability_low": source_probabilities[:, 0],
            "probability_high": source_probabilities[:, 1],
        }
    )
    source_table.to_csv(
        output_dir / "source_propagated_labels.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "source_condition": source_condition,
                "target_condition": target_condition,
                "n_source_cells": len(source),
                "n_target_cells": len(target),
                "low_target_cells": int(np.sum(target_labels == 0)),
                "high_target_cells": int(np.sum(target_labels == 1)),
                "low_source_cells": int(np.sum(source_labels == 0)),
                "high_source_cells": int(np.sum(source_labels == 1)),
                "low_cluster_center": float(cluster_centers[0]),
                "high_cluster_center": float(cluster_centers[1]),
                "silhouette_score": silhouette,
                "mann_whitney_u": float(mann_whitney.statistic),
                "mann_whitney_p": float(mann_whitney.pvalue),
                "cliffs_delta_low_vs_high": delta,
                "baseline_region_auroc": auc,
            }
        ]
    )
    summary.to_csv(
        output_dir / "heterogeneity_summary.csv",
        index=False,
    )

    plt.figure(figsize=(5, 4))
    plt.scatter(
        source_metric,
        barycentric_metric,
        s=14,
        alpha=0.65,
        c=source_labels,
    )
    limits = [
        min(source_metric.min(), barycentric_metric.min()),
        max(source_metric.max(), barycentric_metric.max()),
    ]
    plt.plot(limits, limits, linestyle="--", linewidth=1)
    plt.xlabel("Source metric")
    plt.ylabel("Barycentric target metric")
    plt.tight_layout()
    plt.savefig(
        output_dir / "heterogeneity_joint_scatter.pdf",
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(4, 4))
    plt.boxplot(
        [low_values, high_values],
        labels=["Low", "High"],
        showfliers=False,
    )
    plt.ylabel("Baseline region signal")
    plt.tight_layout()
    plt.savefig(
        output_dir / "baseline_group_comparison.pdf",
        bbox_inches="tight",
    )
    plt.close()

    save_json(
        {
            "spectral_file": str(Path(spectra_path)),
            "source_condition": source_condition,
            "target_condition": target_condition,
            "metric_numerator": metric_numerator,
            "metric_denominator": metric_denominator,
            "baseline_region": baseline_region,
            "random_state": random_state,
        },
        output_dir / "run_metadata.json",
    )


# =========================================================
# Spectral Velocity
# =========================================================


def fit_shared_pca(
    matrices: Mapping[str, np.ndarray],
    path: Sequence[str],
    n_components: int = 2,
) -> Tuple[PCA, Dict[str, np.ndarray]]:
    """Fit one PCA model to all conditions along an ordered path."""
    if n_components < 2:
        raise ValueError("n_components must be at least 2.")

    missing = [condition for condition in path if condition not in matrices]
    if missing:
        raise ValueError(f"Path conditions were not found: {missing}")

    stacked = np.vstack([matrices[condition] for condition in path])
    pca = PCA(n_components=n_components)
    stacked_embedding = pca.fit_transform(stacked)

    embeddings: Dict[str, np.ndarray] = {}
    cursor = 0

    for condition in path:
        n_cells = matrices[condition].shape[0]
        embeddings[condition] = stacked_embedding[cursor:cursor + n_cells]
        cursor += n_cells

    return pca, embeddings


def compute_velocity_step(
    source_spectra: np.ndarray,
    target_spectra: np.ndarray,
    source_embedding: np.ndarray,
    target_embedding: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute one source-level barycentric velocity vector per source cell.
    """
    transport = compute_emd_coupling(source_spectra, target_spectra)
    barycentric_embedding = transport.row_coupling @ target_embedding
    velocity = barycentric_embedding - source_embedding
    magnitude = np.linalg.norm(velocity, axis=1)

    return barycentric_embedding, velocity, magnitude


def run_velocity_analysis(
    spectra_path: str | Path,
    path: Sequence[str],
    output_dir: str | Path,
    cell_id_column: str = "cell_id",
    condition_column: str = "condition",
    n_components: int = 2,
) -> None:
    """Run dose-indexed Spectral Velocity analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table, feature_columns = load_spectral_table(
        spectra_path,
        cell_id_column=cell_id_column,
        condition_column=condition_column,
    )
    matrices = build_condition_matrices(
        table,
        feature_columns,
        condition_column,
    )
    ids = build_condition_ids(
        table,
        cell_id_column,
        condition_column,
    )

    if len(path) < 2:
        raise ValueError("Velocity path must contain at least two conditions.")

    pca, embeddings = fit_shared_pca(
        matrices,
        path,
        n_components=n_components,
    )

    velocity_tables: List[pd.DataFrame] = []

    for step, (source_name, target_name) in enumerate(
        zip(path[:-1], path[1:])
    ):
        barycentric_embedding, velocity, magnitude = compute_velocity_step(
            matrices[source_name],
            matrices[target_name],
            embeddings[source_name],
            embeddings[target_name],
        )

        step_table = pd.DataFrame(
            {
                "step": step,
                "source_condition": source_name,
                "target_condition": target_name,
                "source_cell_id": ids[source_name],
                "source_pc1": embeddings[source_name][:, 0],
                "source_pc2": embeddings[source_name][:, 1],
                "barycentric_pc1": barycentric_embedding[:, 0],
                "barycentric_pc2": barycentric_embedding[:, 1],
                "velocity_pc1": velocity[:, 0],
                "velocity_pc2": velocity[:, 1],
                "velocity_magnitude": magnitude,
            }
        )

        step_table.to_csv(
            output_dir / f"velocity_step_{step}.csv",
            index=False,
        )
        velocity_tables.append(step_table)

    all_velocity = pd.concat(velocity_tables, ignore_index=True)
    all_velocity.to_csv(
        output_dir / "spectral_velocity_all_steps.csv",
        index=False,
    )

    plt.figure(figsize=(6, 5))

    for condition in path:
        embedding = embeddings[condition]
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            s=10,
            alpha=0.5,
            label=condition,
        )

    for step_table in velocity_tables:
        plt.quiver(
            step_table["source_pc1"],
            step_table["source_pc2"],
            step_table["velocity_pc1"],
            step_table["velocity_pc2"],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.002,
            alpha=0.35,
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(
        output_dir / "spectral_velocity.pdf",
        bbox_inches="tight",
    )
    plt.close()

    save_json(
        {
            "spectral_file": str(Path(spectra_path)),
            "path": list(path),
            "n_components": n_components,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        },
        output_dir / "run_metadata.json",
    )


# =========================================================
# Combination-path analysis
# =========================================================


def parse_path_argument(value: str) -> List[str]:
    """Parse one comma-separated path."""
    path = [item.strip() for item in value.split(",") if item.strip()]

    if len(path) < 2:
        raise argparse.ArgumentTypeError(
            "Each path must contain at least two comma-separated conditions."
        )

    return path


def compute_path_results(
    matrices: Mapping[str, np.ndarray],
    path: Sequence[str],
) -> Tuple[List[Dict], List[np.ndarray]]:
    """Compute adjacent-step VIP-OT perturbations along one path."""
    missing = [condition for condition in path if condition not in matrices]
    if missing:
        raise ValueError(f"Combination-path conditions were not found: {missing}")

    summaries: List[Dict] = []
    perturbations: List[np.ndarray] = []

    for step, (source_name, target_name) in enumerate(
        zip(path[:-1], path[1:])
    ):
        source = matrices[source_name]
        target = matrices[target_name]

        transport = compute_emd_coupling(source, target)
        result = compute_perturbation_vectors(
            source,
            target,
            transport.row_coupling,
        )

        summaries.append(
            {
                "step": step,
                "source_condition": source_name,
                "target_condition": target_name,
                "n_source_cells": source.shape[0],
                "n_target_cells": target.shape[0],
                "mean_velocity_magnitude": float(
                    np.linalg.norm(
                        result.perturbation_vectors,
                        axis=1,
                    ).mean()
                ),
            }
        )
        perturbations.append(result.perturbation_vectors)

    return summaries, perturbations


def run_combination_analysis(
    spectra_path: str | Path,
    paths: Sequence[Sequence[str]],
    output_dir: str | Path,
    cell_id_column: str = "cell_id",
    condition_column: str = "condition",
    regions: Sequence[Tuple[str, Tuple[float, float]]] | None = None,
) -> None:
    """Run path-dependent drug-combination perturbation analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table, feature_columns = load_spectral_table(
        spectra_path,
        cell_id_column=cell_id_column,
        condition_column=condition_column,
    )
    matrices = build_condition_matrices(
        table,
        feature_columns,
        condition_column,
    )

    wavenumbers = (
        parse_numeric_wavenumbers(feature_columns)
        if regions
        else None
    )

    summary_rows: List[Dict] = []
    mean_dfsp_rows: List[pd.DataFrame] = []
    region_rows: List[Dict] = []

    for path_index, path in enumerate(paths):
        path_name = f"path_{path_index + 1}"
        summaries, perturbations = compute_path_results(
            matrices,
            path,
        )

        for summary, delta in zip(summaries, perturbations):
            summary_rows.append(
                {
                    "path": path_name,
                    "path_definition": " -> ".join(path),
                    **summary,
                }
            )

            mean_dfsp = delta.mean(axis=0)
            mean_dfsp_table = pd.DataFrame(
                {
                    "path": path_name,
                    "path_definition": " -> ".join(path),
                    "step": summary["step"],
                    "source_condition": summary["source_condition"],
                    "target_condition": summary["target_condition"],
                    "feature": feature_columns,
                    "mean_dfsp": mean_dfsp,
                }
            )
            mean_dfsp_rows.append(mean_dfsp_table)

            if regions and wavenumbers is not None:
                for region_name, interval in regions:
                    mask = spectral_region_mask(
                        wavenumbers,
                        interval,
                    )
                    per_cell_region = delta[:, mask].mean(axis=1)

                    region_rows.append(
                        {
                            "path": path_name,
                            "path_definition": " -> ".join(path),
                            "step": summary["step"],
                            "source_condition": summary["source_condition"],
                            "target_condition": summary["target_condition"],
                            "region": region_name,
                            "region_start": interval[0],
                            "region_stop": interval[1],
                            "mean_change": float(per_cell_region.mean()),
                            "median_change": float(np.median(per_cell_region)),
                            "std_change": float(
                                per_cell_region.std(ddof=1)
                            ),
                        }
                    )

            safe_source = summary["source_condition"].replace("/", "_")
            safe_target = summary["target_condition"].replace("/", "_")
            np.savez_compressed(
                output_dir
                / f"{path_name}_step_{summary['step']}_{safe_source}_to_{safe_target}.npz",
                perturbation_vectors=delta,
                mean_dfsp=mean_dfsp,
            )

    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(
        output_dir / "combination_path_summary.csv",
        index=False,
    )

    mean_dfsp_table = pd.concat(
        mean_dfsp_rows,
        ignore_index=True,
    )
    mean_dfsp_table.to_csv(
        output_dir / "combination_path_mean_dfsp.csv",
        index=False,
    )

    if region_rows:
        pd.DataFrame(region_rows).to_csv(
            output_dir / "combination_path_region_summary.csv",
            index=False,
        )

    plt.figure(figsize=(7, 4))

    for (path_name, step), group in mean_dfsp_table.groupby(
        ["path", "step"],
        sort=False,
    ):
        x_values = (
            np.asarray([float(value) for value in group["feature"]])
            if all(
                _is_float_string(value)
                for value in group["feature"]
            )
            else np.arange(len(group))
        )

        plt.plot(
            x_values,
            group["mean_dfsp"],
            label=f"{path_name}, step {step}",
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel(
        "Wavenumber"
        if all(_is_float_string(value) for value in feature_columns)
        else "Feature index"
    )
    plt.ylabel("Mean perturbation")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(
        output_dir / "combination_path_dfsp.pdf",
        bbox_inches="tight",
    )
    plt.close()

    save_json(
        {
            "spectral_file": str(Path(spectra_path)),
            "paths": [list(path) for path in paths],
            "regions": [
                {"name": name, "interval": interval}
                for name, interval in (regions or [])
            ],
        },
        output_dir / "run_metadata.json",
    )


def _is_float_string(value: str) -> bool:
    """Return True if a string can be converted to float."""
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def parse_region_argument(
    value: str,
) -> Tuple[str, Tuple[float, float]]:
    """Parse a region formatted as 'name,start,stop'."""
    parts = [item.strip() for item in value.split(",")]

    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Regions must be formatted as 'name,start,stop'."
        )

    name = parts[0]

    try:
        start = float(parts[1])
        stop = float(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Region start and stop must be numeric."
        ) from exc

    if start >= stop:
        raise argparse.ArgumentTypeError(
            "Region start must be smaller than stop."
        )

    return name, (start, stop)


# =========================================================
# Command-line interface
# =========================================================


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run representative downstream VIP-OT analyses."
    )
    subparsers = parser.add_subparsers(
        dest="analysis",
        required=True,
    )

    heterogeneity = subparsers.add_parser(
        "heterogeneity",
        help="Run baseline-dependent heterogeneity analysis.",
    )
    heterogeneity.add_argument("--spectra", required=True)
    heterogeneity.add_argument("--source-condition", required=True)
    heterogeneity.add_argument("--target-condition", required=True)
    heterogeneity.add_argument(
        "--metric-numerator",
        required=True,
        type=parse_interval,
        help="Numerator interval formatted as start,stop.",
    )
    heterogeneity.add_argument(
        "--metric-denominator",
        required=True,
        type=parse_interval,
        help="Denominator interval formatted as start,stop.",
    )
    heterogeneity.add_argument(
        "--baseline-region",
        required=True,
        type=parse_interval,
        help="Baseline comparison interval formatted as start,stop.",
    )
    heterogeneity.add_argument("--output", required=True)
    heterogeneity.add_argument(
        "--cell-id-column",
        default="cell_id",
    )
    heterogeneity.add_argument(
        "--condition-column",
        default="condition",
    )
    heterogeneity.add_argument(
        "--random-state",
        type=int,
        default=0,
    )

    velocity = subparsers.add_parser(
        "velocity",
        help="Run dose-indexed Spectral Velocity analysis.",
    )
    velocity.add_argument("--spectra", required=True)
    velocity.add_argument(
        "--path",
        required=True,
        type=parse_path_argument,
        help="Ordered comma-separated condition path.",
    )
    velocity.add_argument("--output", required=True)
    velocity.add_argument(
        "--cell-id-column",
        default="cell_id",
    )
    velocity.add_argument(
        "--condition-column",
        default="condition",
    )
    velocity.add_argument(
        "--n-components",
        type=int,
        default=2,
    )

    combination = subparsers.add_parser(
        "combination",
        help="Run path-dependent combination analysis.",
    )
    combination.add_argument("--spectra", required=True)
    combination.add_argument(
        "--path",
        action="append",
        required=True,
        type=parse_path_argument,
        help=(
            "Comma-separated condition path. Repeat --path to analyze "
            "multiple routes."
        ),
    )
    combination.add_argument(
        "--region",
        action="append",
        type=parse_region_argument,
        default=None,
        help=(
            "Optional spectral region formatted as name,start,stop. "
            "Repeat for multiple regions."
        ),
    )
    combination.add_argument("--output", required=True)
    combination.add_argument(
        "--cell-id-column",
        default="cell_id",
    )
    combination.add_argument(
        "--condition-column",
        default="condition",
    )

    return parser


def main() -> None:
    """Command-line entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.analysis == "heterogeneity":
        run_heterogeneity_analysis(
            spectra_path=args.spectra,
            source_condition=args.source_condition,
            target_condition=args.target_condition,
            metric_numerator=args.metric_numerator,
            metric_denominator=args.metric_denominator,
            baseline_region=args.baseline_region,
            output_dir=args.output,
            cell_id_column=args.cell_id_column,
            condition_column=args.condition_column,
            random_state=args.random_state,
        )

    elif args.analysis == "velocity":
        run_velocity_analysis(
            spectra_path=args.spectra,
            path=args.path,
            output_dir=args.output,
            cell_id_column=args.cell_id_column,
            condition_column=args.condition_column,
            n_components=args.n_components,
        )

    elif args.analysis == "combination":
        run_combination_analysis(
            spectra_path=args.spectra,
            paths=args.path,
            output_dir=args.output,
            cell_id_column=args.cell_id_column,
            condition_column=args.condition_column,
            regions=args.region,
        )

    else:
        raise RuntimeError(f"Unknown analysis: {args.analysis}")

    print(
        f"{args.analysis.capitalize()} analysis completed. "
        f"Results written to: {Path(args.output).resolve()}"
    )


if __name__ == "__main__":
    main()
