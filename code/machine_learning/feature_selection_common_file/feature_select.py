"""
Feature selection pipeline using Random Forest-based Boruta algorithm.

This script performs:
1. Zero-variance feature removal.
2. Boruta feature importance ranking using a balanced RandomForest classifier.
3. Selection of top-K most relevant features.

The selected features are saved to a text file and reused consistently
across all downstream classification models to ensure experimental fairness
and reproducibility.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

# CONFIG
SEED = 42
TOPK = 100
N_JOBS_BORUTA = 8 

# ARGUMENTS 
if len(sys.argv) != 4:
    print("Usage: python feature_select.py <x_train.csv> <y_train.csv> <output_prefix>")
    sys.exit(1)

x_train_path = sys.argv[1]
y_train_path = sys.argv[2]
out_prefix   = sys.argv[3]

# LOAD DATA 
x_train = pd.read_csv(x_train_path)
y_train = pd.read_csv(y_train_path).squeeze()

# ZERO VARIANCE
var_mask = x_train.var(axis=0) > 0
x_train = x_train.loc[:, var_mask]

print(f"After zero-variance removal: {x_train.shape}", flush=True)

# BORUTA 
rf = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=N_JOBS_BORUTA
)

boruta = BorutaPy(
    estimator=rf,
    n_estimators="auto",
    random_state=SEED
)

boruta.fit(x_train.values, y_train.values)

ranking = boruta.ranking_
selected_idx = np.argsort(ranking)[:min(TOPK, len(ranking))]
selected_features = x_train.columns[selected_idx]

# SAVE FEATURES 
feature_file = out_prefix + "_selected_features.txt"

with open(feature_file, "w") as f:
    for feat in selected_features:
        f.write(feat + "\n")

print(f"Saved {len(selected_features)} features to {feature_file}", flush=True)
