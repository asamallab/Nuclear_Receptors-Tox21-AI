"""
SMILES-based classification using MoLFormer-XL with focal loss for
class imbalance handling and and PR-threshold optimization.
"""

import sys
import torch
import numpy as np
import pandas as pd
import time
import os

start = time.time()
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from transformers import set_seed
set_seed(42)

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
import torch.nn.functional as F


# HPC CONTROL
num_cores = int(os.environ.get('OMP_NUM_THREADS', os.cpu_count()))
torch.set_num_threads(num_cores)
torch.set_num_interop_threads(num_cores)

# Enable CPU optimizations
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')

# ARGUMENTS
if len(sys.argv) != 7:
    print(
        "Usage: python3 molformer_edc_optimized.py "
        "<train_csv> <val_csv> <test_csv> <epoch_log> <output_txt> <model_path>"
    )
    sys.exit(1)

train_path = sys.argv[1]
val_path   = sys.argv[2]
test_path  = sys.argv[3]
epoch_log  = sys.argv[4]
output_txt = sys.argv[5]
MODEL_NAME = sys.argv[6] 

# LOAD DATA
train = pd.read_csv(train_path)
val   = pd.read_csv(val_path)
test  = pd.read_csv(test_path)

LABEL_COL = "activity_status"

# ENSURE LABELS ARE INTEGERS
train[LABEL_COL] = train[LABEL_COL].astype(int)
val[LABEL_COL]   = val[LABEL_COL].astype(int)
test[LABEL_COL]  = test[LABEL_COL].astype(int)

# CLASS DISTRIBUTION
train_class_counts = train[LABEL_COL].value_counts().sort_index()
print(f"\nTraining set class distribution:")
for cls, count in train_class_counts.items():
    print(f"  Class {cls}: {count} ({count/len(train)*100:.2f}%)")

# Convert to HF Dataset
train_ds = Dataset.from_pandas(train)
val_ds   = Dataset.from_pandas(val)
test_ds  = Dataset.from_pandas(test)

# Load MoLFormer
print("\n" + "="*60)
print("Loading MoLFormer model...")
print("="*60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    trust_remote_code=True,
    local_files_only=True
)

print(f"Model loaded: {MODEL_NAME}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# TOKENIZATION 
def tokenize(batch):
    return tokenizer(
        batch["SMILES"],
        padding="max_length",
        truncation=True,
        max_length=256
    )
    
num_proc = min(num_cores, 8)  # Cap at 8 to avoid overhead

train_ds = train_ds.map(tokenize, batched=True, num_proc=num_proc, desc="Tokenizing train")
val_ds   = val_ds.map(tokenize, batched=True, num_proc=num_proc, desc="Tokenizing val")
test_ds  = test_ds.map(tokenize, batched=True, num_proc=num_proc, desc="Tokenizing test")

train_ds = train_ds.rename_column(LABEL_COL, "labels")
val_ds   = val_ds.rename_column(LABEL_COL, "labels")
test_ds  = test_ds.rename_column(LABEL_COL, "labels")

train_ds.set_format("torch")
val_ds.set_format("torch")
test_ds.set_format("torch")

print("Tokenization complete!")

# FOCAL LOSS CONFIGURATION
labels_np = train[LABEL_COL].values
class_counts = np.bincount(labels_np)

alpha = torch.tensor(
    len(labels_np) / (2.0 * class_counts),
    dtype=torch.float
)

print("\n" + "="*60)
print("Focal Loss Configuration")
print("="*60)
print(f"Alpha weights: {alpha.tolist()}")
print(f"Gamma: 2.0")

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            at = self.alpha[targets]
            ce_loss = at * ce_loss

        loss = ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean()

class FocalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = FocalLoss(alpha=alpha.to(logits.device), gamma=2)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss
    
# EVAL METRICS
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "mcc": matthews_corrcoef(labels, preds),
        "roc_auc": roc_auc_score(labels, probs),
        "auc_pr":average_precision_score(labels,probs)
    }

# TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir = output_txt[:-4] + "_molformer_epo_50",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # Increased for CPU efficiency
    per_device_eval_batch_size=32,   # Even larger for evaluation
    gradient_accumulation_steps=2,
    num_train_epochs=50,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
    
    # HPC-specific optimizations
    dataloader_num_workers=4,  # Parallel data loading
    dataloader_pin_memory=False,  # CPU-only, no GPU pinning needed
    fp16=False,  # CPU training, no mixed precision
    disable_tqdm=False,  # Show progress bars
    report_to="none",  # Disable wandb/tensorboard for HPC
    log_level="info",
    logging_first_step=True,
    save_safetensors=False
)

print(f"Batch size (train):       {training_args.per_device_train_batch_size}")
print(f"Batch size (eval):        {training_args.per_device_eval_batch_size}")
print(f"Gradient accumulation:    {training_args.gradient_accumulation_steps}")
print(f"Effective batch size:     {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Learning rate:            {training_args.learning_rate}")
print(f"Epochs:                   {training_args.num_train_epochs}")
print(f"Dataloader workers:       {training_args.dataloader_num_workers}")

# SAVE EPOCH METRICS CALLBACK
class SaveMetricsCallback(TrainerCallback):
    def __init__(self):
        self.best_f1 = 0.0
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            with open(epoch_log, "a") as f:
                f.write(f"\nEpoch {int(state.epoch)}\n")
                f.write("----------------------------\n")
                for k, v in metrics.items():
                    f.write(f"{k} : {v}\n")
            
            eval_f1 = metrics.get('eval_f1', 0)
            if eval_f1 > self.best_f1:
                self.best_f1 = eval_f1
                print(f"\n New best F1 score: {self.best_f1:.4f} at epoch {int(state.epoch)}")
    
    def on_train_begin(self, args, state, control, **kwargs):
        # Clear previous epoch log
        with open(epoch_log, "w") as f:
            f.write("MoLFormer Training - Epoch Metrics Log\n")
            f.write("=" * 60 + "\n")

# TRAINING 
trainer = FocalTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[
        SaveMetricsCallback()
    ]
)
trainer.train()

print("\n" + "="*60)
print("Training Complete!")

# Threshold Optimization
print("\n" + "="*60)
print("Optimizing classification threshold on validation set...")
print("="*60)

val_preds = trainer.predict(val_ds)
val_logits = val_preds.predictions
val_probs = torch.softmax(torch.tensor(val_logits), dim=1)[:, 1].numpy()
val_true = val[LABEL_COL].values

pr_precision, pr_recall, thresholds = precision_recall_curve(val_true, val_probs)
f1_scores = 2 * (pr_precision[:-1] * pr_recall[:-1]) / \
            (pr_precision[:-1] + pr_recall[:-1] + 1e-8)

best_threshold = thresholds[np.argmax(f1_scores)]
best_f1_val = np.max(f1_scores)

print(f"Optimal threshold: {best_threshold:.4f}")
print(f"Validation F1 at optimal threshold: {best_f1_val:.4f}")

# Final Test Evaluation
print("\n" + "="*60)
print("Evaluating on test set...")
print("="*60)

test_preds = trainer.predict(test_ds)
logits = test_preds.predictions
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
auc = roc_auc_score(y_true, probs)
auc_pr = average_precision_score(y_true, probs)

end = time.time() - start

# PRINT RESULTS
print("\n" + "="*60)
print("TEST SET RESULTS")
print("="*60)
print(f"Best Threshold : {best_threshold:.4f}")
print(f"Accuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1-score       : {f1:.4f}")
print(f"MCC            : {mcc:.4f}")
print(f"ROC-AUC        : {auc:.4f}")
print(f"AUC-PR         : {auc_pr:.4f}")
print(f"\nConfusion Matrix:")
print(f"  TN={tn}, FP={fp}")
print(f"  FN={fn}, TP={tp}")
print(f"\nTime taken: {end:.2f} seconds ({end/60:.2f} minutes)")
print("="*60)

# SAVE RESULTS
with open(output_txt, "w") as f:
    f.write("MoLFormer EDC Classification Results\n")
    f.write("="*60 + "\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Training samples: {len(train)}\n")
    f.write(f"Validation samples: {len(val)}\n")
    f.write(f"Test samples: {len(test)}\n")
    f.write("\n")
    f.write("Test Set Performance:\n")
    f.write("-"*60 + "\n")
    f.write(f"Best Threshold : {best_threshold:.4f}\n")
    f.write(f"Accuracy       : {accuracy:.4f}\n")
    f.write(f"Precision      : {precision:.4f}\n")
    f.write(f"Recall         : {recall:.4f}\n")
    f.write(f"F1-score       : {f1:.4f}\n")
    f.write(f"MCC            : {mcc:.4f}\n")
    f.write(f"ROC-AUC        : {auc:.4f}\n")
    f.write(f"AUC-PR         : {auc_pr:.4f}\n")
    f.write("\n")
    f.write("Confusion Matrix:\n")
    f.write(f"  True Negatives  (TN) = {tn}\n")
    f.write(f"  False Positives (FP) = {fp}\n")
    f.write(f"  False Negatives (FN) = {fn}\n")
    f.write(f"  True Positives  (TP) = {tp}\n")
    f.write("\n")
    f.write(f"Total training time: {end:.2f} seconds ({end/60:.2f} minutes)\n")
    f.write("="*60 + "\n")

print("\nTraining complete!")
