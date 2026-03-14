# ChemBERTa SMILES-Based Classification Model

## Step 1: Prepare Input Files

Each file must contain:

-   `SMILES` column\
-   `activity_status` column (0/1 labels)

------------------------------------------------------------------------

## Step 2: Run the Model

Use the following command:

``` bash
python3 chemberta.py <train_csv> <val_csv> <test_csv> <epoch_log> <output_txt>
```

### Example

``` bash
python3 /path/chemberta.py /path/train.csv /path/val.csv /path/test.csv /path/epoch_metrics.txt /path/chemberta_results.txt
```

------------------------------------------------------------------------

## Output Files

After running the script, the following files will be generated:

-   `models/..._model/` → Best fine-tuned ChemBERTa model (based on
    validation F1)
-   `epoch_metrics.txt` → Per-epoch validation metrics
-   `chemberta_results.txt` → Final test metrics, optimized threshold,
    confusion matrix, and total runtime
