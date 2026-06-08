# VIP-OT: Dissecting Single-Cell Biochemical State Dynamics under Perturbation via Vibrational Painting and Optimal Transport

Official implementation for the paper:

**“VIP-OT: Dissecting Single-Cell Biochemical State Dynamics under Perturbation via Vibrational Painting and Optimal Transport”**

**Python:** 3.10+  
**License:** CC BY-NC 4.0

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

## Conceptual workflow

1. Acquire fixed-cell FTIR or Raman measurements under control and perturbed conditions.
2. Extract one vibrational spectrum per cell.
3. Represent the control and perturbed populations as distributions in a shared spectral space.
4. Compute a source-to-target optimal-transport coupling.
5. Row-normalize the coupling to obtain source-conditioned target weights.
6. Estimate a barycentric target state for each source cell.
7. Define the perturbation vector as:

\[
\Delta_i = \sum_j \widetilde{\gamma}_{ij} y_j - x_i
\]

where \(x_i\) is the source-cell spectrum, \(y_j\) is a target-cell spectrum, and \(\widetilde{\gamma}_{ij}\) is the row-normalized transport weight.

The resulting perturbation vectors form the computational basis for the downstream analyses in this repository.

---

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
│
├── examples/
│   └── demo_vipot.py
│
├── data/
│   ├── README.md
│   ├── control_demo.csv
│   ├── anisomycin_demo.csv
│   ├── spikein_source.csv
│   ├── spikein_target.csv
│   ├── drug_panel_demo.csv
│   ├── moa_metadata.csv
│   ├── heterogeneity_demo.csv
│   ├── dose_series_demo.csv
│   └── combination_demo.csv
│
├── results/
└── figures/
```

The demonstration data files may contain only a representative subset of the conditions analyzed in the paper.

---

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

## Input data

The analysis scripts operate on **preprocessed single-cell vibrational spectra**.

Each row represents one cell, and each spectral column represents the signal measured at one spectral feature or wavenumber.

A minimal source or target table has the following format:

```text
cell_id, feature_1, feature_2, feature_3, ...
cell_0001, ...
cell_0002, ...
```

For analyses involving experimental conditions:

```text
cell_id, condition, feature_1, feature_2, ...
cell_0001, control, ...
cell_0002, anisomycin, ...
```

For the spike-in identity benchmark:

```text
cell_id, cell_line, feature_1, feature_2, ...
cell_0001, 231, ...
cell_0002, HT29, ...
```

For analyses using spectral intervals, the spectral-column names should be numeric wavenumbers, for example:

```text
cell_id, condition, 1000.0, 1003.8, 1007.7, ...
```

### Preprocessing scope

Raw-image reconstruction, cell segmentation, spectral quality control, instrument-specific correction, and other platform-dependent preprocessing steps are not included in this repository.

These procedures depend on the imaging modality, acquisition configuration, and laboratory workflow. The demonstration matrices are provided after the preprocessing procedures described in the manuscript Methods.

---

## Quick start

### Minimal VIP-OT inference

The following example computes exact balanced OT between a source and target population and saves the barycentric target spectra, cell-level perturbation vectors, mean DFSP, and coupling matrix.

```bash
python examples/demo_vipot.py \
    --source data/control_demo.csv \
    --target data/anisomycin_demo.csv \
    --output results/demo_anisomycin
```

Outputs:

```text
results/demo_anisomycin/
├── coupling.npz
├── barycentric_target.csv
├── perturbation_vectors.csv
├── mean_dfsp.csv
└── run_metadata.json
```

---

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
| `demo_vipot.py` | Minimal OT inference example |

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

## Data availability

Representative preprocessed demonstration data are included in the `data/` directory.

The complete datasets supporting the study are available from the corresponding authors upon reasonable request, subject to applicable institutional and data-sharing requirements.

---

## License

This repository is distributed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

Use of the code and demonstration materials for commercial purposes is not permitted without prior authorization from the copyright holders.

---

## Citation

A formal citation will be added upon publication.

```bibtex
@article{vipot,
  title   = {VIP-OT: Dissecting Single-Cell Biochemical State Dynamics under Perturbation via Vibrational Painting and Optimal Transport},
  author  = {Li, Xuemeng and others},
  journal = {To be updated},
  year    = {To be updated}
}
```

---

## Contact

For questions regarding the repository or data access, please contact the corresponding authors listed in the manuscript.
