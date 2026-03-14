# Fingerprint-Based Classification Model

## How to Run

Use the following command:

``` bash
python3 model.py <x_train.csv> <y_train.csv> <x_test.csv> <y_test.csv> <output_prefix>
```

### Example ( only fingerprint data)

``` bash
python3 /path/model.py /path/train_X.csv /path/train_y.csv /path/test_X.csv /path/test_y.csv /path/model_name_results
```

## Output Files

After running the script, the following files will be generated:

-   `model_name_results.pkl` → Trained model file
-   `model_name_results.txt` → Evaluation metrics, and predictions


