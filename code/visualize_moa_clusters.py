# File: visualize_moa_clusters.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import umap
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D
import argparse


def visualize_clusters(results_root, output_figure):
    """
    Loads perturbation vectors from different drug experiments, performs UMAP,
    and saves a plot where cells are colored by their drug's MoA.
    """
    print("Starting MoA cluster visualization...")

    # A simplified mapping of drugs to their Mechanism of Action (MoA)
    # This helps in coloring the plot logically.
    drug_to_moa = {
        'anisomycin': 'Protein Synthesis', 'cycloheximide': 'Protein Synthesis', 'emetine': 'Protein Synthesis',
        'bortezomib': 'Protein Degradation', 'mg-132': 'Protein Degradation',
        'doxorubicin': 'DNA Intercalation', 'daunorubicin': 'DNA Intercalation', 'epirubicin': 'DNA Intercalation',
        'tvb3166': 'FASN Inhibitor', 'triacsin-c': 'ACS Inhibitor',
        'dactolisib': 'mTOR/PI3K Inhibitor', 'everolimus': 'mTOR/PI3K Inhibitor',
        'gefitinib': 'EGFR Inhibitor',
        'taxol': 'Microtubule Stabilization', 'vincristine': 'Microtubule Stabilization'
    }

    # Define a color palette for each MoA for consistent plotting
    moa_colors = {
        'Protein Synthesis': '#fb4570',
        'Protein Degradation': '#ffa384',
        'DNA Intercalation': '#104210',
        'FASN Inhibitor': '#74bdcb',
        'ACS Inhibitor': '#756153',
        'mTOR/PI3K Inhibitor': '#fec84d',
        'EGFR Inhibitor': '#a4e8e0',
        'Microtubule Stabilization': '#b9a3d1'
    }

    all_vectors = []
    all_labels = []

    # Loop through each subdirectory in the results root folder
    for drug_name in os.listdir(results_root):
        drug_dir = os.path.join(results_root, drug_name)
        if not os.path.isdir(drug_dir):
            continue

        # This is the file that our Optimal Transport script will create
        vector_file = os.path.join(drug_dir, 'perturbation_vectors.csv')

        if os.path.exists(vector_file):
            print(f"Loading perturbation vectors for {drug_name}...")
            df_vectors = pd.read_csv(vector_file, index_col=0)

            # Subsample to keep the plot clean (e.g., max 500 cells per drug)
            if len(df_vectors) > 500:
                df_vectors = df_vectors.sample(n=500, random_state=42)

            all_vectors.append(df_vectors.values)
            all_labels.extend([drug_name] * len(df_vectors))
        else:
            print(f"Warning: Could not find perturbation vectors for {drug_name}. Skipping.")

    if not all_vectors:
        print("Error: No perturbation vectors found. Exiting.")
        return

    # Combine all data into single arrays
    X = np.vstack(all_vectors)
    y_labels = np.array(all_labels)

    # --- UMAP Analysis ---
    print("Performing UMAP dimensionality reduction...")
    # 1. Scale the data
    X_scaled = StandardScaler().fit_transform(X)

    # 2. Apply UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    # --- Plotting ---
    print("Generating the plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))

    # Get the MoA for each data point
    moa_labels = [drug_to_moa.get(drug, 'Unknown') for drug in y_labels]
    unique_moas = sorted(list(set(moa_labels)))

    # Plot each MoA as a group for a clean legend
    for moa in unique_moas:
        if moa == 'Unknown': continue

        idx = [i for i, label in enumerate(moa_labels) if label == moa]
        ax.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            c=[moa_colors.get(moa, '#808080')],
            s=5,
            alpha=0.7,
            label=moa
        )

    ax.set_title('Single-Cell Perturbation Vectors Group by MoA', fontsize=16, fontweight='bold')
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Create a clean, readable legend
    legend = ax.legend(title='Mechanism of Action', fontsize=10, markerscale=3)
    plt.setp(legend.get_title(), fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save the final figure
    fig.savefig(output_figure, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Successfully saved UMAP plot to: {output_figure}")


if __name__ == '__main__':
    # This allows the script to be run from the command line, as described in the README
    parser = argparse.ArgumentParser(description='Visualize MoA clusters from perturbation vectors.')
    parser.add_argument('--results_root', type=str, required=True,
                        help='Root directory containing the results of OT inference for each drug.')
    parser.add_argument('--output_figure', type=str, required=True, help='Path to save the final UMAP plot PNG file.')

    args = parser.parse_args()

    visualize_clusters(args.results_root, args.output_figure)