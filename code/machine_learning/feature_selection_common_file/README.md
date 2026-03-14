# Feature Selection Pipeline (Boruta + Random Forest)

This script performs feature selection using a Random Forest-based Boruta algorithm.

## How to Run

Use the following command:

``` bash
python3 feature_select.py <x_train.csv> <y_train.csv> <output_prefix>
```

### Example

``` bash
python3 /path/feature_select.py /path/train_X.csv /path/train_y.csv /path/feature_selection_output
```
------------------------------------------------------------------------

## Output File

After running the script, the following file will be generated:

-   `feature_selection_output_selected_features.txt` → Contains selected
    feature names (one per line)

This file should be passed to downstream classification models.

------------------------------------------------------------------------
