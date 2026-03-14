# FEATURE FILE 

## Overview

This directory contains molecular feature files for **8430 chemicals**, identified by their DSSTox substance IDs (`dsstox_substance_id`). All files share `dsstox_substance_id` as the primary key and can be joined across feature sets.

- **RDKit** — ECFP4, FCFP4, Layered, MACCS fingerprints, 2D and 3D descriptors
- **PaDEL-Descriptor v2.21** — 2D and 3D descriptors

All files are compressed in 7-Zip (.7z) format. Extract using any compatible tool before use.

---

## Files

### [`ECFP4_8430_chemicals.csv.7z`](./ECFP4_8430_chemicals.csv.7z)
ECFP4 fingerprint (radius 2) from RDKit comprising **1024 binary bits**.

### [`FCFP4_8430_chemicals.csv.7z`](./FCFP4_8430_chemicals.csv.7z)
FCFP4 fingerprint (radius 2) from RDKit comprising **1024 binary bits**.

### [`Layered_8430_chemicals.csv.7z`](./Layered_8430_chemicals.csv.7z)
Layered fingerprint from RDKit comprising **2048 binary bits**.

### [`MACCS_8430_chemicals.csv.7z`](./MACCS_8430_chemicals.csv.7z)
MACCS structural keys from RDKit comprising **167 binary bits**.

### [`PaDEL_2D_and_3D_8430_chemicals.csv.7z`](./PaDEL_2D_and_3D_8430_chemicals.csv.7z)
**1875 2D and 3D descriptors** computed using PaDEL-Descriptor v2.21. 

### [`RDKit_2D_and_3D_8430_chemicals.csv.7z`](./RDKit_2D_and_3D_8430_chemicals.csv.7z)
**231 2D and 3D descriptors** computed using RDKit.

### [`SMILES_8430_chemicals.csv.7z`](./SMILES_8430_chemicals.csv.7z)
Canonical SMILES strings for all 8430 chemicals computed using Open Babel v3.1.0.