""" Descriptor-based classification using SVM with SMOTE balancing and 10-fold Stratified Cross-Validation.
Features were pre-selected using feature_select.py and consistently applied across all models."""

import sys
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix,average_precision_score
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# CONFIG
SEED = 42

# Change based on your system specifications
n_jobs_grid_search = -1  

# ARGUMENTS
if len(sys.argv) != 7:
    print("Usage: python3 model.py <x_train> <y_train> <x_test> <y_test> <features_file> <output_file>")
    sys.exit(1)

x_train_path, y_train_path, x_test_path, y_test_path, features_file, output_path = sys.argv[1:]

# LOAD DATA 
x_train = pd.read_csv(x_train_path)
y_train = pd.read_csv(y_train_path).squeeze()
x_test  = pd.read_csv(x_test_path)
y_test  = pd.read_csv(y_test_path).squeeze()

# LOAD SELECTED FEATURES
with open(features_file) as f:
    selected_features = [line.strip() for line in f]

x_train = x_train[selected_features]
x_test  = x_test[selected_features]

print(f"Using {len(selected_features)} selected features", flush=True)

# PIPELINE 
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=SEED, k_neighbors=3)),
    ("model", SVC(
        probability=True,
        random_state=SEED
    ))
])

# PARAM GRID
param_grid = {
    "model__kernel": ["linear", "poly", "rbf", "sigmoid"],
    "model__class_weight": ["balanced"],
    "model__random_state": [SEED],
    "model__gamma": ["auto", "scale"],
    "model__probability": [True]
}

# CV
cv = RepeatedStratifiedKFold(
    n_splits=10,
    n_repeats=1,
    random_state=SEED
)

# GRID SEARCH
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=n_jobs_grid_search,
    verbose=1
)

grid.fit(x_train, y_train)
best_model = grid.best_estimator_
with open(output_path + ".pkl", "wb") as f:
    pickle.dump(best_model, f)
print("Best params:", grid.best_params_)

# TEST EVALUATION 
y_pred = best_model.predict(x_test)
y_prob = best_model.predict_proba(x_test)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
mcc       = matthews_corrcoef(y_test, y_pred)
auc       = roc_auc_score(y_test, y_prob)
auc_pr    = average_precision_score(y_test, y_prob)

# PRINT RESULTS
print("\n===== TEST RESULTS =====")
print(f"Threshold : 0.5000")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"MCC       : {mcc:.4f}")
print(f"ROC-AUC   : {auc:.4f}")
print(f"AUC-PR    : {auc_pr:.4f}")
print("Confusion Matrix:")
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# SAVE RESULTS
with open(output_path + '.txt', "w") as f:
    f.write("===== TEST RESULTS (Fixed 0.5 Threshold) =====\n")
    f.write("Threshold  : 0.5000\n\n")
    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1-score  : {f1:.4f}\n")
    f.write(f"MCC       : {mcc:.4f}\n")
    f.write(f"ROC-AUC   : {auc:.4f}\n")
    f.write(f"AUC-PR    : {auc_pr:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}\n\n")
    f.write("Best Parameter Settings\n")
    f.write(f"{grid.best_params_}\n")
    f.write("\nPredictions:\n")
    f.write("----------------\n")
    f.write("y_pred\ty_prob\n")
    for pred, prob in zip(y_pred, y_prob):
        f.write(f"{pred}\t{prob:.6f}\n")
