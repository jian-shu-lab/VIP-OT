# File: run_ot_inference.py

import pandas as pd
import numpy as np
import ot  # This is the Python Optimal Transport (POT) library
import os
import argparse


def run_ot_inference(control_path, treated_path, output_dir):
    """
    Computes the Optimal Transport coupling between control and treated cell spectra
    and saves the inferred perturbation vectors.
    """
    print(f"--- Running OT Inference for {os.path.basename(treated_path)} ---")

    # --- 1. Load Data ---
    try:
        print(f"Loading control data from: {control_path}")
        df_control = pd.read_csv(control_path, index_col=0)

        print(f"Loading treated data from: {treated_path}")
        df_treated = pd.read_csv(treated_path, index_col=0)

        # Ensure data is in numpy array format for OT calculation
        control_spectra = df_control.values
        treated_spectra = df_treated.values

    except FileNotFoundError as e:
        print(f"Error: {e}. Please check your file paths.")
        return

    # --- 2. Compute Cost Matrix ---
    # The cost matrix represents the "distance" between every control cell and every treated cell.
    # Here, we use the squared Euclidean distance, a standard choice.
    print("Computing cost matrix between cell populations...")
    cost_matrix = ot.dist(control_spectra, treated_spectra, metric='sqeuclidean')

    # Normalize the cost matrix to prevent numerical issues
    cost_matrix /= cost_matrix.max()

    # --- 3. Solve the Optimal Transport Problem ---
    # We assume uniform weights for all cells in each population.
    num_control = control_spectra.shape[0]
    num_treated = treated_spectra.shape[0]

    a = np.ones((num_control,)) / num_control
    b = np.ones((num_treated,)) / num_treated

    print("Solving the OT problem using Earth Mover's Distance (EMD)...")
    # ot.emd() is the exact solver for the OT problem.
    coupling_matrix = ot.emd(a, b, cost_matrix)

    # --- 4. Calculate Perturbation Vectors ---
    # For each control cell, we find its "average" corresponding treated cell
    # based on the weights in the coupling matrix.
    print("Calculating perturbation vectors...")

    # We normalize each row of the coupling matrix to sum to 1 to get probabilities
    # This gives us the probability of a control cell matching any given treated cell.
    # We add a small epsilon to avoid division by zero if a control cell has no couplings.
    row_sums = coupling_matrix.sum(axis=1, keepdims=True)
    prob_matrix = coupling_matrix / (row_sums + 1e-9)

    # The expected treated spectra for each control cell is the weighted average
    # of all treated spectra, using the probabilities as weights.
    expected_treated_spectra = prob_matrix @ treated_spectra

    # The perturbation vector is simply the difference.
    perturbation_vectors = expected_treated_spectra - control_spectra

    # --- 5. Save Results ---
    # Save the vectors to a CSV file for the visualization script to use.
    df_vectors = pd.DataFrame(perturbation_vectors, index=df_control.index, columns=df_control.columns)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'perturbation_vectors.csv')

    df_vectors.to_csv(output_path)
    print(f"✅ Successfully saved perturbation vectors to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Optimal Transport inference between control and treated spectra.')
    parser.add_argument('--control_path', type=str, required=True, help='Path to the control group data CSV file.')
    parser.add_argument('--treated_path', type=str, required=True, help='Path to the treated group data CSV file.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the output perturbation vectors.')

    args = parser.parse_args()

    run_ot_inference(args.control_path, args.treated_path, args.output_dir)