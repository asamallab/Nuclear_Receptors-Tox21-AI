# MOLRAG_LLAMA-Based Molecular Activity Classification

Prompt-based classification using [LLAMA 3.1 8B (GGUF)](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct)

------------------------------------------------------------------------

## Step 1: Generate Prompts

Before running the model, generate the prompt file using [`molrag_llama_generate_prompts.py`](./molrag_llama_generate_prompts.py):


``` bash
python3 molrag_llama_generate_prompts.py
```

This will generate:

    prompts.pkl

------------------------------------------------------------------------

## Step 2: Run the MOLRAG_LLAMA Model

Use the following command:

``` bash
python3 molrag_llama_model.py <model_path> <prompt_file> <output_dir>
```

### Example

``` bash
python3 /path/molrag_llama_model.py /path/model.gguf /path/prompts.pkl /path/output_predictions
```

------------------------------------------------------------------------

## Step 3: Generate Final Evaluation Results

After predictions are generated, evaluate the results and create the
final output files using [`molrag_llama_generate_output_text_file.py`](./molrag_llama_generate_output_text_file.py):

``` bash
python3 molrag_llama_generate_output_text_file.py <receptors> <endpoints> <val_preds_dir> <val_gt_dir> <test_preds_dir> <test_gt_dir> <output_dir>
```

### Example

``` bash
python3 /path/molrag_llama_generate_output_text_file.py ER,AR agonist,antagonist /path/val_preds /path/val_gt /path/test_preds /path/test_gt /path/final_results
```

------------------------------------------------------------------------

## Output Files

After running the scripts, the following files will be generated:

-   `predictions.csv` → Model prediction probabilities and predicted
    labels
-   `predictions.json` → JSON version of predictions
-   `run_metadata.json` → Model run metadata (model path, threads,
    threshold, timestamp)
-   `<receptor>_<endpoint>_llama_final.txt` → Final evaluation metrics
    including ROC-AUC, AUPRC, accuracy, precision, recall, F1-score,
    MCC, confusion matrix, and optimized threshold
