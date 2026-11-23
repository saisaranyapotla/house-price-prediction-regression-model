# House Price Prediction — Regression Project

A reproducible, end-to-end supervised regression project for estimating residential property prices. It includes a local CSV dataset, leakage-safe preprocessing, train/test evaluation, cross-validation, model comparison, tuning, charts, and a written discussion of assumptions and limitations.

## Dataset

`data/housing.csv` contains 2,400 **synthetic, illustrative** housing records. It is included so the project runs without downloading data or requiring an API key. Features include floor area, bedrooms, bathrooms, property age, distance to city, school rating, garage, neighborhood, and condition. Some predictor values are intentionally missing to exercise imputation.

**Important:** This is a demonstration dataset, not verified market transaction data. Do not use its predictions for real property valuation. To use a real dataset, supply a CSV with a numeric target column (default: `price`) and run `--data path/to/your.csv --target your_target`.

## Requirements

- Python 3.9+
- Packages in `requirements.txt`

## Quick start

### Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/train.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/train.py
```

For a faster smoke run:

```bash
python src/train.py --quick
```

## What the program does

1. Loads the CSV and removes rows with a missing target.
2. Splits data into training and held-out test sets (80/20, fixed random seed).
3. Detects numeric and categorical columns.
4. Imputes numeric values with training-fold medians and categorical values with training-fold most-frequent values.
5. Scales numeric features and one-hot encodes categorical features. Unknown categories are handled safely.
6. Compares a median `DummyRegressor` baseline, Linear Regression, Ridge Regression, a Decision Tree, and a tuned Random Forest.
7. Uses shuffled 5-fold cross-validation on the training set; tunes Random Forest hyperparameters using `GridSearchCV`.
8. Evaluates the selected model once on the held-out test set with RMSE, MAE, and R².
9. Writes an actual-vs-predicted scatter plot, permutation feature-importance chart, CSV tables, and a JSON summary.

## Outputs

After running, the `outputs/` folder contains:

- `model_comparison.csv` — CV RMSE and held-out test RMSE, MAE, R² for each candidate.
- `actual_vs_predicted.png` — scatter plot with the ideal prediction diagonal.
- `feature_importance.png` — permutation feature importance chart.
- `feature_importance.csv` — feature importance means and variation.
- `summary.json` — best model, metrics, selected forest parameters, split sizes, and leakage-prevention note.

## Methodology and leakage prevention

All preprocessing is part of each model's scikit-learn `Pipeline` / `ColumnTransformer`. Imputers, scaling, and category vocabulary are learned only from the training portion of each cross-validation fold, then applied to its validation fold. The final test set is not used for fitting or model selection. Permutation importance is computed on the held-out test features after model selection. The target column is excluded from the feature matrix.

## Metrics

- **RMSE**: square root of mean squared error; penalizes large errors more strongly and is expressed in the target's units.
- **MAE**: average absolute prediction error, also in target units.
- **R²**: proportion of target variance explained relative to predicting the test-set mean; can be negative.

Lower RMSE/MAE is better; higher R² is better. Compare the candidate models with the baseline, not just with each other. A useful model should improve materially over the baseline on held-out data.

## Assumptions, limitations, and failure cases

- The bundled data is synthetic and encodes an illustrative relationship between property attributes and price. Its performance is not evidence of real-world accuracy.
- Synthetic feature distributions and price relationships may not represent a particular city, time period, housing segment, or current market.
- The random split assumes records are independent and identically distributed. For real data with dates, repeated properties, or geographic clusters, use time-based or group-based validation to avoid optimistic results.
- Missingness is imputed, not modeled; if data is missing systematically, this can introduce bias.
- One-hot encoding does not capture geographic distance or neighborhood hierarchy beyond the supplied categories.
- Extreme outliers, unobserved renovations, interest rates, local policy, and market shocks can cause large errors.
- Feature importance is associational, can be unstable with correlated features, and is not causal.
- R² can be negative on a difficult test set. RMSE is sensitive to extreme price errors.
- A model trained on one region or period may fail when applied to another. Monitor errors and retrain with representative, recent, legally obtained data.

## Custom data example

```bash
python src/train.py --data data/my_houses.csv --target sale_price --test-size 0.2 --cv 5
```

Your CSV should contain at least 30 rows, a numeric target, and one or more predictor columns. Numeric and non-numeric predictors are handled automatically. Do not include post-sale information or fields that reveal the target.
