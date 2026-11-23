from __future__ import annotations
import argparse
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate house-price regression models.")
    parser.add_argument("--data", default=str(ROOT / "data" / "housing.csv"))
    parser.add_argument("--target", default="price")
    parser.add_argument("--output", default=str(ROOT / "outputs"))
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Use a smaller forest search for faster execution.")
    args = parser.parse_args()

    data_path, out_dir = Path(args.data), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' is not a column in {data_path}")
    df = df.dropna(subset=[args.target]).copy()
    X, y = df.drop(columns=[args.target]), df[args.target]
    if len(df) < 30:
        raise ValueError("At least 30 labeled rows are recommended.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    cv = KFold(n_splits=args.cv, shuffle=True, random_state=42)

    candidates = {
        "Dummy baseline": DummyRegressor(strategy="median"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=42),
    }
    if args.quick:
        forest_grid = {"model__n_estimators": [150], "model__max_depth": [None, 16],
                       "model__min_samples_leaf": [2]}
    else:
        forest_grid = {"model__n_estimators": [150, 300],
                       "model__max_depth": [None, 16],
                       "model__min_samples_leaf": [1, 2]}
    forest_pipe = Pipeline([
        ("preprocess", build_preprocessor(X_train)),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])
    search = GridSearchCV(
        forest_pipe, forest_grid, cv=cv, scoring="neg_root_mean_squared_error",
        n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    candidates["Random Forest (tuned)"] = search.best_estimator_

    rows = []
    fitted = {}
    for name, estimator in candidates.items():
        if name == "Random Forest (tuned)":
            model = estimator
            cv_rmse = -search.best_score_
        else:
            model = Pipeline([("preprocess", build_preprocessor(X_train)), ("model", estimator)])
            scores = cross_val_score(model, X_train, y_train, cv=cv,
                                     scoring="neg_root_mean_squared_error", n_jobs=-1)
            cv_rmse = -scores.mean()
            model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        rows.append({"model": name, "cv_rmse": float(cv_rmse), "test_rmse": rmse,
                     "test_mae": float(mean_absolute_error(y_test, pred)),
                     "test_r2": float(r2_score(y_test, pred))})
        fitted[name] = model

    results = pd.DataFrame(rows).sort_values("test_rmse")
    results.to_csv(out_dir / "model_comparison.csv", index=False)
    best_name = results.iloc[0]["model"]
    best_model = fitted[best_name]
    best_pred = best_model.predict(X_test)

    # Evaluation chart: held-out actual versus predicted values.
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_pred, alpha=0.55)
    low = min(float(y_test.min()), float(np.min(best_pred)))
    high = max(float(y_test.max()), float(np.max(best_pred)))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.title(f"Actual vs predicted — {best_name}")
    plt.tight_layout()
    plt.savefig(out_dir / "actual_vs_predicted.png", dpi=160)
    plt.close()

    # Permutation importance is calculated on the untouched holdout data.
    importance = permutation_importance(
        best_model, X_test, y_test, n_repeats=8, random_state=42,
        scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    imp = pd.DataFrame({"feature": X_test.columns,
                        "importance_mean": importance.importances_mean,
                        "importance_std": importance.importances_std})
    imp = imp.sort_values("importance_mean", ascending=False)
    imp.to_csv(out_dir / "feature_importance.csv", index=False)
    top = imp.head(12).sort_values("importance_mean")
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    plt.xlabel("Decrease in score when feature is shuffled (RMSE-based)")
    plt.title("Permutation feature importance")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=160)
    plt.close()

    summary = {
        "best_model": best_name,
        "best_model_test_metrics": results.iloc[0].to_dict(),
        "random_forest_best_params": search.best_params_,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target": args.target,
        "data_file": str(data_path),
        "note": "All imputing, scaling and encoding are inside sklearn Pipelines fitted on training folds only."
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(results.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Outputs saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
