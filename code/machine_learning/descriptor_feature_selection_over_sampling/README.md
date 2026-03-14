# Descriptor-Based Classification Model

Model with SMOTE balancing (Using pre-selected features)

------------------------------------------------------------------------

## Step 1: Generate Selected Features

Before running the model, generate the selected feature file.

Navigate to the [`feature_selection_common_file`](../feature_selection_common_file/) directory and run:

``` bash
python3 feature_select.py
```

This will generate:

    model_name_selected_features.txt

------------------------------------------------------------------------

## Step 2: Run the Model

Use the following command:

``` bash
python3 model.py <x_train> <y_train> <x_test> <y_test> <features_file> <output_file>
```

### Example

``` bash
python3 /path/model.py /path/train_X.csv /path/train_y.csv /path/test_X.csv /path/test_y.csv /path/selected_features.txt /path/model_name_results
```

------------------------------------------------------------------------

## Output Files

After running the script, the following files will be generated:

-   `model_name_results.pkl` → Trained model (SMOTE + Model pipeline)
-   `model_name_results.txt` → Evaluation metrics, confusion matrix, best parameters, and predictions
