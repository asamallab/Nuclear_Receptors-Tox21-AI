# PROCESSING DATA 

This repository contains **split index files** that map chemical IDs to their train/test/val (validation) assignments across 3 data splits. These are designed to be merged with master feature files to reconstruct training and test sets.

---

## Repository Structure

```
./data/
├── Features/                                    # Master feature files (shared across all receptors)
│   ├── ECFP4_8430_chemicals.csv                 # ECFP4 fingerprints
│   ├── FCFP4_8430_chemicals.csv                 # FCFP4 fingerprints
│   ├── Layered_8430_chemicals.csv               # Layered fingerprints
│   ├── MACCS_8430_chemicals.csv                 # MACCS fingerprints
│   ├── PaDEL_2D_and_3D_8430_chemicals.csv       # PaDEL descriptors
│   ├── RDKit_2D_and_3D_8430_chemicals.csv       # RDKit descriptors
│   └── SMILES_8430_chemicals.csv                # SMILES strings
│
├── DA_index/<train_test_split_#>/                                   # Delta/Gamma/Kappa index files
│   └── <receptor>/<activity>/
│       └── <feature>_da_index.csv
│
└── <receptor>/                                  # e.g. AHR, AR, CAR, ERa ...
    └── <activity>/                              # agonist / antagonist / combined
        ├── descriptor_split_index.csv
        ├── descriptor_fingerprint_split_index.csv
        ├── fingerprint_split_index.csv
        └── smiles_split_index.csv
```

---

## Split Index File Format

Each of the 4 split index files per receptor/activity has this structure:

| Column    | Description |
|-----------|-------------|
| `ID`      | Chemical identifier (e.g. `DTXSID9045637`) |
| `label`   | Activity label (`0` = inactive, `1` = active) |
| `split_1` | Assignment in Split 1: `train`, `test` or `val` |
| `split_2` | Assignment in Split 2: `train`, `test` or `val` |
| `split_3` | Assignment in Split 3: `train`, `test` or `val` |

---

## Feature File — Split Index Mapping

| Split Index File                          | Master Feature File(s) to Merge With |
|-------------------------------------------|--------------------------------------|
| `descriptor_split_index.csv`              | `PaDEL_2D_and_3D_8430_chemicals.csv.7z` and `RDKit_2D_and_3D_8430_chemicals.csv.7z`|
| `descriptor_fingerprint_split_index.csv`  | Both all descriptor file AND all fingerprint file combined |
| `fingerprint_split_index.csv`             | `ECFP4_`, `FCFP4_`, `Layered_`,`MACCS_8430_chemicals.csv.7z` or all fingerprints |
| `smiles_split_index.csv`                  | `SMILES_8430_chemicals.csv.7z`|

---

## How to Get X_train, X_test, y_train, y_test

The split index files do **not** contain feature values — they only tell you which chemical IDs belong to train/test/val in each split. The actual feature values live in the master [`Features/`](./Features/) files. So you always need to combine both.

### Step 1 — Pick your split index file
Choose based on what feature type you want. For example, if you want to train on fingerprints, open `fingerprint_split_index.csv` for your receptor and activity of choice (e.g. `AHR/agonist/fingerprint_split_index.csv`).

### Step 2 — Open the corresponding master feature file
Open the matching file from the [`Features/`](./Features/) folder and get the corresponding files required.

### Step 3 — Join them on the ID column
The split index has an `ID` column and the feature file has a matching ID column (named `dsstox_substance_id` or similar). Match rows from both files using this ID. This gives you a combined table where each row has both the feature values and the train/test/val assignment for that chemical.

### Step 4 — Filter by the split column
The combined table has a `split_1` column (or `split_2` / `split_3` depending on which split you want to work with). Rows where the value is `train` form your training set. Rows where it says `test` form your test set. For SMILES files there is also a `val` split available. Here, rows with `val` form your validation set.

### Step 5 — Separate features from labels
From the training rows, all the feature columns (fingerprint bits, descriptor values, SMILES strings, etc.) become your `X_train`, and the `label` column becomes your `y_train`. Do the same for the test rows to get `X_test` and `y_test`.

---

