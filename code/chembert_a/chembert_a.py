"""
SMILES-based classification using ChemBERTa (seyonec/ChemBERTa-zinc-base-v1)
with class-weighted CrossEntropy loss for imbalance handling

"""

import sys
import torch
import numpy as np
import pandas as pd
import time

start = time.time()
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)

from torch.nn import CrossEntropyLoss

# ARGUMENTS
if len(sys.argv) != 6:
    print(
        "Usage: python3 chemberta_edc.py "
        "<train_csv> <val_csv> <test_csv> <epoch_log> <output_txt>"
    )
    sys.exit(1)

train_path = sys.argv[1]
val_path   = sys.argv[2]
test_path  = sys.argv[3]
epoch_log  = sys.argv[4]
output_txt = sys.argv[5]

train = pd.read_csv(train_path)
val   = pd.read_csv(val_path)
test  = pd.read_csv(test_path)

LABEL_COL = "activity_status"

train_ds = Dataset.from_pandas(train)
val_ds   = Dataset.from_pandas(val)
test_ds  = Dataset.from_pandas(test)

# MODEL LOADING 
MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# TOKENIZATION
def tokenize(batch):
    return tokenizer(
        batch["SMILES"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_ds = train_ds.map(tokenize, batched=True)
val_ds   = val_ds.map(tokenize, batched=True)
test_ds  = test_ds.map(tokenize, batched=True)

train_ds = train_ds.rename_column(LABEL_COL, "labels")
val_ds   = val_ds.rename_column(LABEL_COL, "labels")
test_ds  = test_ds.rename_column(LABEL_COL, "labels")

train_ds.set_format("torch")
val_ds.set_format("torch")
test_ds.set_format("torch")

labels_np = train[LABEL_COL].astype(int).values
class_counts = np.bincount(labels_np)

# CLASS WEIGHT BALANCING 
class_weights = torch.tensor(
    len(labels_np) / (2.0 * class_counts),
    dtype=torch.float
)

# CUSTOM TRAINER TO USE CLASS-WEIGHTED LOSS ( FOCAL LOSS )
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        probs = torch.softmax(logits, dim=1)
        ce_loss = CrossEntropyLoss(reduction="none")(logits, labels)
        pt = probs[torch.arange(len(labels)), labels]
        focal_weight = self.class_weights.to(logits.device)[labels] * (1 - pt) ** 2
        loss = (focal_weight * ce_loss).mean()

        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    auc = roc_auc_score(labels, probs)
    auc_pr = average_precision_score(labels,probs)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mcc": mcc,
        "roc_auc": auc,
        "auc_pr" :auc_pr
    }

# TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir = output_txt.replace("outputs", "models")[:-4] + "_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=50,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=1
)

class SaveMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            with open(epoch_log, "a") as f:
                f.write(f"\nEpoch {int(state.epoch)}\n")
                f.write("----------------------------\n")
                for k, v in metrics.items():
                    f.write(f"{k} : {v}\n")

# TRAINING
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[SaveMetricsCallback()]
)

trainer.class_weights = class_weights
trainer.train()

# EVALUATION

# VALIDATION THRESHOLD OPTIMIZATION 
val_predictions = trainer.predict(val_ds)

val_logits = val_predictions.predictions
val_probs = torch.softmax(torch.tensor(val_logits), dim=1)[:, 1].numpy()
val_true = val[LABEL_COL].values

pr_precision, pr_recall, thresholds = precision_recall_curve(val_true, val_probs)

f1_scores = 2 * (pr_precision[:-1] * pr_recall[:-1]) / \
            (pr_precision[:-1] + pr_recall[:-1] + 1e-8)

best_threshold = thresholds[np.argmax(f1_scores)]

# TEST EVALUATION USING VALIDATION THRESHOLD 
test_predictions = trainer.predict(test_ds)

logits = test_predictions.predictions
probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
y_true = test[LABEL_COL].values

y_pred = (probs >= best_threshold).astype(int)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall    = recall_score(y_true, y_pred, zero_division=0)
f1        = f1_score(y_true, y_pred, zero_division=0)
mcc       = matthews_corrcoef(y_true, y_pred)
auc       = roc_auc_score(y_true, probs)
auc_pr    = average_precision_score(y_true, probs)
end = time.time() - start

with open(output_txt, "w") as f:
    f.write("ChemBERTa EDC Classification Results\n")
    f.write("------------------------------------\n")
    f.write(f"Best Threshold (from validation): {best_threshold:.6f}\n")
    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1-score  : {f1:.4f}\n")
    f.write(f"MCC       : {mcc:.4f}\n")
    f.write(f"ROC-AUC   : {auc:.4f}\n")
    f.write(f"AUC-PR    : {auc_pr:.4f}\n")
    f.write(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}\n")
    f.write(f"Time taken: {end}")

print("Training complete. Results saved.")
