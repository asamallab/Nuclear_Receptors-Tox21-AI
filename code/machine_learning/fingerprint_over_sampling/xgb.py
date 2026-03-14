"""Fingerprint-based classification using XGboost, SMOTE balancing, and 10-fold Stratified CV"""


import pandas as pd
import sys
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix,average_precision_score
)
import pickle
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# ARGUMENTS 
if len(sys.argv) != 6:
    print("Usage: python3 model.py <x_train> <y_train> <x_test> <y_test> <output_file>")
    sys.exit(1)

# LOAD DATA
x_train = pd.read_csv(sys.argv[1])
y_train = pd.read_csv(sys.argv[2]).iloc[:, 0]
x_test  = pd.read_csv(sys.argv[3])
y_test  = pd.read_csv(sys.argv[4]).iloc[:, 0]
text_file_path = sys.argv[5]

# CONFIG
SEED = 42
n_jobs_xgb = 1

# Change based on your system specifications
n_jobs_grid_search = -1

# PIPELINE 
pipeline = Pipeline([
    ("smote", SMOTE(random_state=SEED, k_neighbors=3)),
    ("xgb", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=n_jobs_xgb
    ))
])

# PARAM GRID 
param_grid = {
    "xgb__n_estimators": [300, 500],
    "xgb__max_depth": [3, 5, 7],
    "xgb__learning_rate": [0.01, 0.05, 0.1],
    "xgb__subsample": [0.8, 1.0],
    "xgb__colsample_bytree": [0.8, 1.0]
}

# CV 
cv = RepeatedStratifiedKFold(
    n_splits=10,
    n_repeats=1,
    random_state=SEED
)

# GRID SEARCH
grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=n_jobs_grid_search,
    verbose=1
)

grid.fit(x_train, y_train)
print("Best params:", grid.best_params_)

model = grid.best_estimator_

best_model = grid.best_estimator_
with open(text_file_path + ".pkl", "wb") as f:
    pickle.dump(best_model, f)

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
aucpr = average_precision_score(y_test, y_prob)

# PRINT RESULTS
print("\n===== TEST RESULTS =====")
print(f"Threshold : 0.5000")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"MCC       : {mcc:.4f}")
print(f"ROC-AUC   : {auc:.4f}")
print(f"AUC-PR    : {aucpr:.4f}")
print("Confusion Matrix:")
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# SAVE RESULTS
with open(text_file_path + '.txt', "w") as f:
    f.write("===== TEST RESULTS (Fixed 0.5 Threshold) =====\n")
    f.write("Threshold  : 0.5000\n\n")
    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1-score  : {f1:.4f}\n")
    f.write(f"MCC       : {mcc:.4f}\n")
    f.write(f"ROC-AUC   : {auc:.4f}\n")
    f.write(f"AUC-PR    : {aucpr:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}\n\n")
    f.write("Best Parameter Settings\n")
    f.write(f"{grid.best_params_}\n")
    f.write("\nPredictions:\n")
    f.write("----------------\n")
    f.write("y_pred\ty_prob\n")
    for pred, prob in zip(y_pred, y_prob):
        f.write(f"{pred}\t{prob:.6f}\n")
