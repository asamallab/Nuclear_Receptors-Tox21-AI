# DGCL: Dual GIN-GAT Contrastive Learning for Molecular Bioactivity Prediction

## Step 1: Prepare Input Files

### Activity Files (train / val / test)
Each file must contain:
- `dsstox_substance_id` column
- `SMILES` column
- `activity_status` column (0/1 labels)

------------------------------------------------------------------------

## Step 2: Run the Model

### DGCL with fingerprint data

```bash
python3 dgcl_fingerprints.py <train_csv> <val_csv> <test_csv> <fingerprint_csv> <checkpoint_dir> <output_txt>
```

#### Example

```bash
python3 /path/dgcl_fingerprints.py /path/train.csv /path/val.csv /path/test.csv /path/ecfp4_fingerprints.csv /path/checkpoints /path/dgcl_fp_results.txt
```
------------------------------------------------------------------------

### DGCL with descriptor data

```bash
python3 dgcl_descriptors.py <train_csv> <val_csv> <test_csv> <descriptor_csv> <checkpoint_dir> <output_txt>
```

#### Example

```bash
python3 /path/dgcl_descriptors.py /path/train.csv /path/val.csv /path/test.csv /path/descriptors.csv /path/checkpoints/ /path/dgcl_desc_results.txt
```
------------------------------------------------------------------------

## Output Files

After running the script, the following files will be generated:

- `<checkpoint_dir>/gin_gat_checkpoint.csv` → Per-trial grid search results log (trial number, hyperparameters, validation F1, runtime)
- `<checkpoint_dir>/gin_gat_checkpoint.pt` → Best model checkpoint saved during grid search (auto-resumes if run is interrupted)
- `<checkpoint_dir>/best_dgcl_gin_gat_model.pt` → Final best model weights, hyperparameters, threshold, and test metrics
- `<output_txt>` → Final test set metrics, best threshold, best hyperparameters, and total runtime

------------------------------------------------------------------------
