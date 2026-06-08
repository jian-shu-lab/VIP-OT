# VIP-OT: Dissecting Single-Cell Biochemical State Dynamics under Perturbation via Vibrational Painting and Optimal Transport

**Official Implementation for the paper: "VIP-OT: Dissecting Single-Cell Biochemical State Dynamics under Perturbation via Vibrational Painting and Optimal Transport"**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)
---

## Overview

VIP-OT is an experimental-computational framework that combines multiplexed vibrational imaging with optimal transport (OT) to infer single-cell perturbation responses from unpaired population snapshots.

Vibrational imaging provides high-content biochemical fingerprints of individual cells, but fixed-cell measurements do not preserve natural cell-to-cell correspondences between control and perturbed populations. VIP-OT addresses this limitation by modeling each population as an empirical distribution in spectral space and computing a probabilistic coupling between source and target cells.

The inferred coupling supports several downstream analyses, including:

- single-cell perturbation-vector inference;
- biological identity-preservation benchmarking;
- comparison with alternative matching strategies;
- mechanism-of-action organization;
- retrospective tracing of response heterogeneity;
- dose-indexed Spectral Velocity analysis; and
- path-dependent drug-combination analysis.

This repository provides compact reference implementations of the principal computational procedures used in the study. It is intended to illustrate the analysis logic starting from preprocessed single-cell spectral matrices.

---


## The VIP-OT Framework

The overall workflow of VIP-OT is illustrated below. The process begins with metabolic labeling and drug perturbation, followed by single-cell vibrational imaging (FTIR or Raman). Optimal Transport is then applied to the extracted spectral profiles to infer a probabilistic coupling between control and perturbed cell populations. This coupling forms the computational backbone for all downstream analyses.

<img width="2065" height="1289" alt="Fig1_v2" src="https://github.com/user-attachments/assets/733787c5-ff87-4334-a65a-2c05a1c99ac0" />


## Installation

The code was developed for Python 3.10 and can be installed using `conda`.

```bash
git clone https://github.com/jian-shu-lab/VIP-OT.git
cd VIP-OT

conda env create -f environment.yml
conda activate vip-ot
```

Main dependencies include:

- NumPy
- pandas
- SciPy
- scikit-learn
- matplotlib
- POT (Python Optimal Transport)
- openpyxl

---

## Data

Due to file size limitations, this repository includes a representative demonstration (`demo`) dataset located in the `/data` directory. This demo dataset contains pre-processed single-cell FTIR spectra from MDA-MB-231 human breast adenocarcinoma cells, including:
* A shared pool of control cells.
* Cells treated with a subset of the 16 drugs discussed in the paper.

Each data file is in `.csv` format, where each row corresponds to a single cell and each column represents the absorbance at a specific wavenumber.

The full dataset is available from the authors upon reasonable request.

## Quick Start: Example Usage

Here, we provide a simple example to demonstrate the core functionality of VIP-OT: **inferring single-cell perturbation vectors and clustering them by Mechanism of Action (MoA)**, which reproduces the core finding of Figure 2 in our manuscript.

1.  **Run Optimal Transport to Infer Perturbation Vectors:**
    This script computes the OT coupling between control and drug-treated cells and calculates the corresponding perturbation vector for each cell.
    ```bash
    # Example for Anisomycin
    python code/run_ot_inference.py --control_path data/control_demo.csv --treated_path data/anisomycin_demo.csv --output_dir results/anisomycin

    # Example for Doxorubicin
    python code/run_ot_inference.py --control_path data/control_demo.csv --treated_path data/doxorubicin_demo.csv --output_dir results/doxorubicin
    ```

2.  **Visualize MoA-specific Clustering:**
    After running the inference for all demo drug conditions, this script will aggregate the perturbation vectors, perform dimensionality reduction using UMAP, and generate a plot showing that drugs with the same MoA cluster together.
    ```bash
    python code/visualize_moa_clusters.py --results_root results/ --output_figure figures/moa_umap.png
    ```
    The generated `moa_umap.png` in the `/figures` directory should show clear separation between different drug MoAs, similar to Figure 2a in the paper.


## Repository structure

```text
VIP-OT/
├── README.md
├── LICENSE
├── environment.yml
│
├── code/
│   ├── vipot.py
│   ├── spikein_benchmark.py
│   ├── moa_benchmark.py
│   └── downstream_analysis.py
├── data/
├── results/
└── figures/
```
## Core implementation

### `code/vipot.py`

This file contains the principal computational utilities:

- pairwise cost-matrix construction;
- exact balanced OT using EMD;
- entropically regularized Sinkhorn OT;
- row normalization of transport plans;
- barycentric target estimation;
- perturbation-vector calculation;
- mean DFSP calculation;
- edge-level difference extraction;
- greedy nearest-neighbor matching;
- Hungarian assignment;
- mutual nearest-neighbor matching; and
- pairwise-condition and trajectory iterators.

Example usage:

```python
from vipot import compute_emd_coupling, compute_perturbation_vectors

transport = compute_emd_coupling(
    source_spectra,
    target_spectra,
    metric="sqeuclidean",
)

result = compute_perturbation_vectors(
    source_spectra,
    target_spectra,
    transport.row_coupling,
)

barycentric_target = result.barycentric_target
perturbation_vectors = result.perturbation_vectors
mean_dfsp = result.mean_dfsp
```

---

## Spike-in identity benchmark

### `code/spikein_benchmark.py`

This script compares:

- exact OT-EMD;
- Sinkhorn OT;
- greedy nearest-neighbor matching;
- Hungarian assignment; and
- mutual nearest neighbors.

The benchmark evaluates:

- source-cell coverage;
- same-line identity preservation;
- within-line soft transport mass;
- cross-line transport mass;
- hard argmax purity;
- cell-line flow matrices; and
- target-label permutation null distributions.

Run:

```bash
python code/spikein_benchmark.py \
    --source data/spikein_source.csv \
    --target data/spikein_target.csv \
    --output results/spikein
```

Optional arguments:

```bash
--sinkhorn-reg 0.05
--n-permutations 1000
--random-state 0
--save-couplings
```

Main outputs:

```text
results/spikein/
├── benchmark_summary.csv
├── hard_matching_pairs.csv
├── soft_coupling_per_cell.csv
├── permutation_null.csv
├── identity_flow_matrices.xlsx
└── run_metadata.json
```

---

## Mechanism-of-action benchmark

### `code/moa_benchmark.py`

This script evaluates whether perturbation representations inferred by different matching methods recover mechanism-of-action-level organization.

The spectral input should contain:

```text
cell_id, condition, feature_1, feature_2, ...
```

The MoA metadata table should contain:

```text
condition, moa
anis, Protein synthesis inhibition
cyc, Protein synthesis inhibition
...
```

Run:

```bash
python code/moa_benchmark.py \
    --spectra data/drug_panel_demo.csv \
    --moa data/moa_metadata.csv \
    --control-condition control \
    --output results/moa_benchmark \
    --n-bootstrap 100
```

The script computes:

- cell-level perturbation vectors;
- condition-level mean DFSPs;
- drug-by-drug Pearson correlation matrices;
- mean within-MoA correlation;
- mean between-MoA correlation;
- within-minus-between separation score;
- source-cell coverage; and
- bootstrap robustness.

Main outputs:

```text
results/moa_benchmark/
├── coverage_summary.csv
├── moa_separation_scores.csv
├── bootstrap_scores.csv
├── run_metadata.json
└── correlation_matrices/
    ├── ot_emd_drug_correlation.csv
    ├── sinkhorn_drug_correlation.csv
    ├── greedy_nn_drug_correlation.csv
    ├── hungarian_drug_correlation.csv
    └── mnn_drug_correlation.csv
```

Cell-level perturbation matrices can optionally be saved using:

```bash
--save-cell-deltas
```

---

## Downstream analyses

### `code/downstream_analysis.py`

This script provides three subcommands:

```text
heterogeneity
velocity
combination
```

### 1. Baseline-dependent heterogeneity

The heterogeneity workflow:

1. computes a treated-cell metabolic or spectral metric;
2. separates treated cells into low and high states using k-means;
3. propagates treated-state labels back to source cells using OT weights;
4. compares baseline spectral features between the inferred source groups; and
5. reports silhouette score, Mann–Whitney test, Cliff’s delta, and AUROC.

Run:

```bash
python code/downstream_analysis.py heterogeneity \
    --spectra data/heterogeneity_demo.csv \
    --source-condition control \
    --target-condition anisomycin \
    --metric-numerator 2070,2150 \
    --metric-denominator 2800,3000 \
    --baseline-region 1720,1760 \
    --output results/heterogeneity
```

Main outputs:

```text
results/heterogeneity/
├── treated_state_labels.csv
├── source_propagated_labels.csv
├── heterogeneity_summary.csv
├── heterogeneity_joint_scatter.pdf
├── baseline_group_comparison.pdf
└── run_metadata.json
```

### 2. Spectral Velocity

Spectral Velocity is defined as the OT-derived, source-level displacement between adjacent conditions in a shared embedding.

For each source cell:

\[
v_i = \sum_j \widetilde{\gamma}_{ij} z_j^{\mathrm{target}} - z_i^{\mathrm{source}}
\]

where \(z\) denotes the shared PCA coordinates.

Run:

```bash
python code/downstream_analysis.py velocity \
    --spectra data/dose_series_demo.csv \
    --path control,dose_1,dose_2,dose_3 \
    --output results/velocity
```

Main outputs:

```text
results/velocity/
├── velocity_step_0.csv
├── velocity_step_1.csv
├── velocity_step_2.csv
├── spectral_velocity_all_steps.csv
├── spectral_velocity.pdf
└── run_metadata.json
```

### 3. Drug-combination path analysis

This analysis computes path-specific perturbation vectors between selected single-agent and combination conditions.

Multiple routes can be specified by repeating `--path`.

Run:

```bash
python code/downstream_analysis.py combination \
    --spectra data/combination_demo.csv \
    --path gefitinib,gefitinib_bortezomib \
    --path bortezomib,gefitinib_bortezomib \
    --region protein,1600,1700 \
    --region lipid,1720,1760 \
    --output results/combination
```

Main outputs:

```text
results/combination/
├── combination_path_summary.csv
├── combination_path_mean_dfsp.csv
├── combination_path_region_summary.csv
├── combination_path_dfsp.pdf
├── path-specific perturbation files
└── run_metadata.json
```

---

## Relation to manuscript analyses

| Repository component | Representative manuscript analysis |
|---|---|
| `vipot.py` | Core OT coupling and perturbation-vector inference |
| `spikein_benchmark.py` | Mixed-cell-line identity-preservation benchmark |
| `moa_benchmark.py` | Alternative matching methods and MoA-level organization |
| `downstream_analysis.py heterogeneity` | Baseline-dependent response heterogeneity |
| `downstream_analysis.py velocity` | Dose-indexed Spectral Velocity |
| `downstream_analysis.py combination` | Path-dependent drug-combination analysis |

The scripts provide representative implementations of the main computational analyses. They are not intended as a complete raw-data-to-figure reproduction pipeline.

---

## Reproducibility notes

- Exact OT uses balanced EMD with uniform source and target marginals.
- The default transport cost is squared Euclidean distance.
- Transport plans are row-normalized before barycentric projection.
- Sinkhorn benchmarking uses a default regularization value of `0.05`.
- Randomized analyses expose a `random_state` argument.
- Large cost matrices and coupling matrices are not saved by default.
- Demonstration datasets may be smaller than the complete datasets used in the manuscript.

---
