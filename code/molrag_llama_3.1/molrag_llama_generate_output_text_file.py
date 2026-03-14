""" 
This script evaluates receptor endpoint classification results 
by first optimizing the decision threshold on the validation set to maximize F1-score.
It then applies the best threshold to the test set, computes performance metrics 
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    matthews_corrcoef, average_precision_score
)

# ARGUMENT PARSING

if len(sys.argv) != 8:
    print("Usage:")
    print("python evaluate_results.py "
          "RECEPTORS ENDPOINTS "
          "VAL_PREDS_DIR VAL_GT_DIR "
          "TEST_PREDS_DIR TEST_GT_DIR "
          "OUTPUT_DIR")
    sys.exit(1)

RECEPTORS = sys.argv[1].split(",")
ENDPOINTS = sys.argv[2].split(",")

VAL_PREDS_DIR  = sys.argv[3]
VAL_GT_DIR     = sys.argv[4]
TEST_PREDS_DIR = sys.argv[5]
TEST_GT_DIR    = sys.argv[6]
OUTPUT_DIR     = sys.argv[7]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# MAIN LOOP

for receptor in RECEPTORS:
    for endpoint in ENDPOINTS:

        print(f"\n Processing {receptor} {endpoint}")
        
        val_pred_file = os.path.join(
            VAL_PREDS_DIR, f"{receptor}_{endpoint}_predictions.csv"
        )

        val_gt_file = os.path.join(
            VAL_GT_DIR,
            receptor,
            endpoint,
            "SMILES",
            f"{receptor}_{endpoint}_val_ID.csv"
        )
        test_pred_file = os.path.join(
            TEST_PREDS_DIR, f"{receptor}_{endpoint}_predictions.csv"
        )

        test_gt_file = os.path.join(
            TEST_GT_DIR,
            receptor,
            endpoint,
            "SMILES",
            f"{receptor}_{endpoint}_test_ID.csv"
        )
        if not (os.path.exists(val_pred_file) and
                os.path.exists(val_gt_file) and
                os.path.exists(test_pred_file) and
                os.path.exists(test_gt_file)):
            print("Missing files. Skipping.")
            continue

        # VALIDATION THRESHOLD OPTIMIZATION
        val_preds = pd.read_csv(val_pred_file)
        val_gt = pd.read_csv(val_gt_file)

        if 'activity_status' in val_gt.columns:
            val_gt = val_gt.rename(columns={'activity_status': 'label'})

        val_merged = pd.merge(
            val_preds,
            val_gt[['dsstox_substance_id', 'label']],
            on='dsstox_substance_id'
        )

        if val_merged.empty:
            print("Empty validation merge. Skipping.")
            continue

        y_val_true = val_merged['label'].values
        y_val_prob = val_merged['prob_active'].values

        thresholds = np.linspace(0.01, 0.99, 299)

        best_f1 = -1
        best_threshold = 0.5

        for t in thresholds:
            preds = (y_val_prob >= t).astype(int)
            f1 = f1_score(y_val_true, preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        print(f"Best validation threshold: {best_threshold:.3f}")
        print(f"Best validation F1: {best_f1:.4f}")

        # TEST EVALUATION USING FIXED THRESHOLD
        test_preds = pd.read_csv(test_pred_file)
        test_gt = pd.read_csv(test_gt_file)

        if 'activity_status' in test_gt.columns:
            test_gt = test_gt.rename(columns={'activity_status': 'label'})

        test_merged = pd.merge(
            test_preds,
            test_gt[['dsstox_substance_id', 'label']],
            on='dsstox_substance_id'
        )

        if test_merged.empty:
            print("Empty test merge. Skipping.")
            continue

        y_test_true = test_merged['label'].values
        y_test_prob = test_merged['prob_active'].values

        y_test_pred = (y_test_prob >= best_threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test_true, y_test_pred).ravel()

        accuracy = accuracy_score(y_test_true, y_test_pred)
        precision = precision_score(y_test_true, y_test_pred, zero_division=0)
        recall = recall_score(y_test_true, y_test_pred, zero_division=0)
        f1 = f1_score(y_test_true, y_test_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test_true, y_test_pred)
        roc_auc = roc_auc_score(y_test_true, y_test_prob)
        auprc = average_precision_score(y_test_true, y_test_prob)

        output_file = os.path.join(
            OUTPUT_DIR,
            f"{receptor}_{endpoint}_llama_final.txt"
        )

        # SAVE RESULTS 
        with open(output_file, "w") as f:
            f.write("==========================================\n")
            f.write(f"LLAMA 3.1 8B - {receptor} {endpoint} TEST PERFORMANCE\n")
            f.write("==========================================\n")
            f.write(f"Total Evaluated : {len(y_test_true)}\n")
            f.write(f"Validation Threshold Used: {best_threshold:.3f}\n")
            f.write("------------------------------------------\n")
            f.write(f"ROC-AUC Score  : {roc_auc:.4f}\n")
            f.write(f"AUC-PR Score   : {auprc:.4f}\n")
            f.write(f"Accuracy       : {accuracy:.4f}\n")
            f.write(f"Precision      : {precision:.4f}\n")
            f.write(f"Recall         : {recall:.4f}\n")
            f.write(f"F1-score       : {f1:.4f}\n")
            f.write(f"MCC Score      : {mcc:.4f}\n")
            f.write("------------------------------------------\n")
            f.write(f"True Positives (TP) : {tp}\n")
            f.write(f"True Negatives (TN) : {tn}\n")
            f.write(f"False Positives(FP) : {fp}\n")
            f.write(f"False Negatives(FN) : {fn}\n")
            f.write("==========================================\n")

        print(f"Saved: {output_file}")

print("\nAll receptors completed successfully.")