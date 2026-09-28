"""Fit OLS, Ridge, and Lasso against price and write findings to output/.

Reuses the same cleaning/feature-engineering steps as analyze.py, kept as a
separate script (and separate output file) so PCA and regression stay two
independent views of the data rather than one conflated model.

Re-run whenever data/airbnb_listing.csv changes:
    python regress.py
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from analyze import (
    DATA_PATH,
    OUTPUT_DIR,
    build_feature_matrix,
    clean_price,
    impute_grouped_median,
    parse_bathrooms,
)

TEST_SIZE = 0.2
RANDOM_STATE = 42
RIDGE_ALPHAS = np.logspace(-2, 3, 30)


def load_and_clean() -> pd.DataFrame:
    raw = pd.read_csv(DATA_PATH)
    raw["id"] = raw["id"].astype("int64")
    df, _ = clean_price(raw)

    df["bathrooms"] = df["bathrooms_text"].apply(parse_bathrooms)
    df["bathrooms"] = impute_grouped_median(df, "bathrooms", "room_type")
    for col in ("bedrooms", "beds"):
        df[col] = impute_grouped_median(df, col, "room_type")
    return df


def build_regression_inputs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x_all, feature_columns = build_feature_matrix(df)
    price_idx = feature_columns.index("price_log")
    x = np.delete(x_all, price_idx, axis=1)
    feature_names = [c for i, c in enumerate(feature_columns) if i != price_idx]
    y = df["price_log"].to_numpy()
    return x, y, feature_names


def coefficient_table(feature_names: list[str], coefficients: np.ndarray) -> list[dict]:
    rows = [{"feature": f, "coefficient": float(c)} for f, c in zip(feature_names, coefficients)]
    rows.sort(key=lambda r: abs(r["coefficient"]), reverse=True)
    return rows


def evaluate(model, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    predictions = model.predict(x_test)
    return {
        "test_r2": float(r2_score(y_test, predictions)),
        "test_rmse": float(root_mean_squared_error(y_test, predictions)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_and_clean()
    x, y, feature_names = build_regression_inputs(df)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    ols = LinearRegression().fit(x_train_scaled, y_train)
    ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(x_train_scaled, y_train)
    lasso = LassoCV(cv=5, random_state=RANDOM_STATE, n_jobs=-1).fit(x_train_scaled, y_train)

    models = {
        "ols": {
            **evaluate(ols, x_test_scaled, y_test),
            "intercept": float(ols.intercept_),
            "coefficients": coefficient_table(feature_names, ols.coef_),
        },
        "ridge": {
            "alpha": float(ridge.alpha_),
            **evaluate(ridge, x_test_scaled, y_test),
            "intercept": float(ridge.intercept_),
            "coefficients": coefficient_table(feature_names, ridge.coef_),
        },
        "lasso": {
            "alpha": float(lasso.alpha_),
            **evaluate(lasso, x_test_scaled, y_test),
            "intercept": float(lasso.intercept_),
            "n_features_zeroed": int((lasso.coef_ == 0).sum()),
            "coefficients": coefficient_table(feature_names, lasso.coef_),
        },
    }

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_rows_used": len(df),
        "target": "price_log",
        "note": (
            "Coefficients are on standardized features fit on the training split, so magnitude is "
            "directly comparable across features and reflects each feature's effect holding the "
            "others constant. This is a separate model from the PCA in analyze.py, not PCA-derived."
        ),
        "train_test_split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_train": len(x_train),
            "n_test": len(x_test),
        },
        "feature_columns_used": feature_names,
        "models": models,
    }

    (OUTPUT_DIR / "regression_results.json").write_text(json.dumps(results, indent=2))

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "ols_coef": ols.coef_,
            "ridge_coef": ridge.coef_,
            "lasso_coef": lasso.coef_,
        }
    )
    coef_df.to_csv(OUTPUT_DIR / "regression_coefficients.csv", index=False)

    print(
        f"fit OLS/Ridge/Lasso on {len(x_train)} train / {len(x_test)} test rows, "
        f"wrote output/regression_results.json, output/regression_coefficients.csv"
    )


if __name__ == "__main__":
    main()
