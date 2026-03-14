""" Fingerprint-based classification using Logistic Regression, SMOTE balancing, and 10-fold Stratified CV"""


import sys
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    average_precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# CONFIG
SEED = 42

# Change based on your system specifications
n_jobs_grid_search = -1

# ARGUMENT CHECK
if len(sys.argv) != 6:
    print("Usage: python3 model.py <x_train> <y_train> <x_test> <y_test> <output_file>")
    sys.exit(1)

x_train_path = sys.argv[1]
y_train_path = sys.argv[2]
x_test_path  = sys.argv[3]
y_test_path  = sys.argv[4]
text_file_path = sys.argv[5]

# LOAD DATA
x_train = pd.read_csv(x_train_path)
y_train = pd.read_csv(y_train_path).squeeze()
x_test  = pd.read_csv(x_test_path)
y_test  = pd.read_csv(y_test_path).squeeze()

# PIPELINE 
pipeline = Pipeline([
    ("smote", SMOTE(random_state=SEED,k_neighbors=3)),
    ("lr", LogisticRegression(max_iter=1000, random_state=SEED))
])

# PARAM GRID
param_grid = {
    "lr__C": [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        1, 2, 3, 4, 5, 7, 9, 11, 15, 20, 25, 30, 35, 40, 50, 100
    ],
    "lr__penalty": ["l1", "l2"],
    "lr__solver": ["liblinear", "saga"]
}

# CV STRATEGY
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
    verbose=1,
    refit=True
)

grid.fit(x_train, y_train)

print("Best score:", grid.best_score_)
print("Best params:", grid.best_params_)

model = grid.best_estimator_
best_model = grid.best_estimator_
with open(text_file_path + ".pkl", "wb") as f:
    pickle.dump(best_model, f)

#TEST EVALUATION
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
