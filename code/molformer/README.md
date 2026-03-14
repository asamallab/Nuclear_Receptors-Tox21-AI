# MoLFormer-XL SMILES-Based Classification Model
------------------------------------------------------------------------

## Step 1: Prepare Input Files

Each file must contain:

-   `SMILES` column\
-   `activity_status` column (0/1 labels)

------------------------------------------------------------------------

## Step 2: Model Path Configuration

This script uses the following [MoLFormer-XL model](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct):

MODEL_NAME = "/models--ibm--MoLFormer-XL-both-10pct/snapshots/7b12d946c181a37f6012b9dc3b002275de070314"

Ensure this local path is provided as the last argument (`model_name`) when
running the script.

------------------------------------------------------------------------

## Step 3: Run the Model

Use the following command:

``` bash
python3 molformer.py <train_csv> <val_csv> <test_csv> <epoch_log> <output_txt> <model_name>
```

### Example

``` bash
python3 /path/molformer.py /path/train.csv /path/val.csv /path/test.csv /path/epoch_metrics.txt /path/molformer_results.txt /path/local_model_path
```

------------------------------------------------------------------------

## Output Files

After running the script, the following files will be generated:

-   `<output_txt>_molformer_epo_50/` → Best fine-tuned MoLFormer model
-   `epoch_metrics.txt` → Per-epoch validation metrics
-   `molformer_results.txt` → Final test metrics, optimized threshold,
    confusion matrix, and total runtime
