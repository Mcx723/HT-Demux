import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import argparse
from pathlib import Path

def evaluate_demux(pred_path: str, truth_path: str):
    """
    Evaluates demultiplexing performance against ground truth.
    """
    if not Path(pred_path).exists() or not Path(truth_path).exists():
        print(f"Error: Files not found. \nPred: {pred_path}\nTruth: {truth_path}")
        return

    # 1. Load Data
    pred_df = pd.read_csv(pred_path)
    truth_df = pd.read_csv(truth_path)

    # 2. Harmonize Ground Truth Labels
    # Based on the user's ground_truth format:
    # - n_cells == 0 -> Negative
    # - n_cells == 1 -> Use true_donors (e.g., HTO_sim_03)
    # - n_cells > 1 or is_doublet -> Multiplet
    def parse_truth(row):
        if row['n_cells'] == 0 or pd.isna(row['true_donors']):
            return "Negative"
        if row['is_doublet'] == True or row['n_cells'] > 1:
            return "Multiplet"
        return str(row['true_donors']).strip()

    truth_df['truth_label'] = truth_df.apply(parse_truth, axis=1)

    # 3. Merge Results
    # Join on barcode (Prediction) and droplet_id (Truth)
    merged = pd.merge(
        pred_df, 
        truth_df[['droplet_id', 'truth_label']], 
        left_on='barcode', 
        right_on='droplet_id', 
        how='inner'
    )

    if merged.empty:
        print("Warning: No matching barcodes found. Check if barcode formats match.")
        return

    # 4. Generate Metrics
    # We use 'assignment_final' which includes 'Unassigned' (below threshold)
    y_true = merged['truth_label']
    y_pred = merged['assignment_final']

    # Get all unique labels involved to define labels for the report
    all_labels = sorted(list(set(y_true.unique()) | set(y_pred.unique())))

    print("\n" + "="*50)
    print("      HT-Demux Performance Evaluation")
    print("="*50)
    print(f"Total droplets matched: {len(merged)}")
    
    # Calculate Overall Accuracy
    accuracy = (y_true == y_pred).sum() / len(merged)
    print(f"Overall Accuracy: {accuracy:.2%}")

    # Detailed Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0))

    # Optional: Breakdown of Unassigned cells
    unassigned_count = (y_pred == "Unassigned").sum()
    if unassigned_count > 0:
        print(f"Note: {unassigned_count} cells were 'Unassigned' due to low confidence.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Demux results")
    parser.add_argument("--pred", type=str, default="results/demux_results.csv", 
                        help="Path to HT-Demux output CSV")
    parser.add_argument("--truth", type=str, default="mydata/ground_truth_droplet.csv", 
                        help="Path to ground truth CSV")
    
    args = parser.parse_args()
    evaluate_demux(args.pred, args.truth)