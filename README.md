# VIP-OT: Dissecting Single-Cell Biochemical State Dynamics under Perturbation via Vibrational Painting and Optimal Transport

**Official Implementation for the paper: "VIP-OT: Dissecting Single-Cell Biochemical State Dynamics under Perturbation via Vibrational Painting and Optimal Transport"**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This repository provides the official implementation and a demonstration dataset for **Vibrational Painting-Optimal Transport (VIP-OT)**. VIP-OT is an integrated experimental-computational framework that couples multiplexed vibrational imaging with optimal transport (OT) to computationally reconstruct single-cell perturbation trajectories from unpaired population snapshots. A central challenge in single-cell biology is modeling how individual cells transition between different states, a task for which traditional vibrational imaging is limited as it only provides static snapshots. VIP-OT overcomes this fundamental barrier by reframing the problem from a distributional perspective, using OT to infer the most likely pairings between control and treated cell populations.

This framework enables a suite of downstream analyses, including tracing drug response heterogeneity back to baseline metabolic states, predicting the post-treatment state of individual cells, and resolving drug combination effects into distinct molecular routes.

## The VIP-OT Framework

The overall workflow of VIP-OT is illustrated below. The process begins with metabolic labeling and drug perturbation, followed by single-cell vibrational imaging (FTIR or Raman). Optimal Transport is then applied to the extracted spectral profiles to infer a probabilistic coupling between control and perturbed cell populations. This coupling forms the computational backbone for all downstream analyses.

<img width="2065" height="1289" alt="Fig1_v2" src="https://github.com/user-attachments/assets/733787c5-ff87-4334-a65a-2c05a1c99ac0" />


## Installation

This code has been tested on a Linux environment (Ubuntu 20.04) with Python 3.10. We recommend using `conda` to manage dependencies and create a reproducible environment.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/jian-shu-lab/VIP-OT.git]
    cd VIP-OT
    ```

2.  **Create and activate the conda environment:**
    All required packages are listed in the `environment.yml` file.
    ```bash
    conda env create -f environment.yml
    conda activate vip-ot
    ```

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

## Repository Structure
```
├── code/ # Source code for analysis
│ ├── run_ot_inference.py
│ └── visualize_moa_clusters.py
├── data/ # Demo datasets
│ ├── control_demo.csv
│ └── Anisomycin_demo.csv
│ └──Doxorubicin_demo.csv
├── figures/ # Generated figures
├── results/ # Intermediate results
├── environment.yml # Conda environment file
├── LICENSE # MIT License
└── README.md # Project documentation
```
