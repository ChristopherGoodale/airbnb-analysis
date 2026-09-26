"""Clean the raw listing data, run PCA, and write findings to output/.

Re-run this whenever data/airbnb_listing.csv changes:
    python analyze.py
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/airbnb_listing.csv")
OUTPUT_DIR = Path("output")

PRICE_CORR_THRESHOLD = 0.3

NUMERIC_FEATURES = [
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "number_of_reviews",
    "latitude",
    "longitude",
    "price_log",
]
CATEGORICAL_FEATURES = ["room_type", "neighbourhood_group"]


def clean_price(df: pd.DataFrame) -> pd.DataFrame:
    price_clean = (
        df["price"].str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip().astype(float)
    )
    df = df.assign(price_clean=price_clean)
    n_zero = int((df["price_clean"] == 0).sum())
    df = df[df["price_clean"] > 0].copy()
    df["price_log"] = np.log1p(df["price_clean"])
    return df, n_zero


def parse_bathrooms(text) -> float:
    if pd.isna(text):
        return np.nan
    t = text.strip().lower()
    if "half-bath" in t:
        return 0.5
    m = re.match(r"(\d+\.?\d*)", t)
    return float(m.group(1)) if m else np.nan


def impute_grouped_median(df: pd.DataFrame, column: str, group_col: str) -> pd.Series:
    filled = df[column].fillna(df.groupby(group_col)[column].transform("median"))
    filled = filled.fillna(df[column].median())
    return filled


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    dummies = pd.get_dummies(df[CATEGORICAL_FEATURES], drop_first=True)
    feature_columns = NUMERIC_FEATURES + list(dummies.columns)
    matrix = pd.concat([df[NUMERIC_FEATURES], dummies], axis=1)[feature_columns]
    return matrix.to_numpy(dtype=float), feature_columns


def correlate_components_with_price(scores: np.ndarray, x_scaled: np.ndarray, price_idx: int) -> list[float]:
    price_col = x_scaled[:, price_idx]
    return [float(np.corrcoef(scores[:, i], price_col)[0, 1]) for i in range(scores.shape[1])]


def rank_features(
    pca: PCA, feature_columns: list[str], price_idx: int, selected: list[int]
) -> list[dict]:
    weights = pca.explained_variance_ratio_[selected]
    weights = weights / weights.sum()

    rankings = []
    for j, feature in enumerate(feature_columns):
        if j == price_idx:
            continue
        loadings_by_component = {f"PC{i + 1}": float(pca.components_[i, j]) for i in selected}
        score = sum(w * abs(pca.components_[i, j]) for w, i in zip(weights, selected))
        rankings.append(
            {
                "feature": feature,
                "importance_score": float(score),
                "loadings_by_component": loadings_by_component,
            }
        )
    rankings.sort(key=lambda r: r["importance_score"], reverse=True)
    return rankings


def compute_borough_concentration(df: pd.DataFrame) -> list[dict]:
    counts = df["neighbourhood_group"].value_counts()
    pct = (counts / counts.sum() * 100).round(2)
    return [
        {"neighbourhood_group": borough, "listing_count": int(counts[borough]), "pct_of_total": float(pct[borough])}
        for borough in counts.index
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    raw = pd.read_csv(DATA_PATH)
    n_rows_raw = len(raw)
    raw["id"] = raw["id"].astype("int64")

    df, n_price_zero = clean_price(raw)

    df["bathrooms"] = df["bathrooms_text"].apply(parse_bathrooms)
    df["bathrooms_imputed"] = df["bathrooms"].isna()
    df["bathrooms"] = impute_grouped_median(df, "bathrooms", "room_type")

    for col in ("bedrooms", "beds"):
        df[f"{col}_imputed"] = df[col].isna()
        df[col] = impute_grouped_median(df, col, "room_type")

    x, feature_columns = build_feature_matrix(df)
    x_scaled = StandardScaler().fit_transform(x)

    pca = PCA(n_components=len(feature_columns))
    scores = pca.fit_transform(x_scaled)

    price_idx = feature_columns.index("price_log")
    price_corr = correlate_components_with_price(scores, x_scaled, price_idx)

    selected = [i for i, c in enumerate(price_corr) if abs(c) >= PRICE_CORR_THRESHOLD]
    if not selected:
        selected = sorted(range(len(price_corr)), key=lambda i: abs(price_corr[i]), reverse=True)[:2]

    feature_ranking = rank_features(pca, feature_columns, price_idx, selected)
    borough_concentration = compute_borough_concentration(df)

    explained = pca.explained_variance_ratio_.tolist()
    cumulative = np.cumsum(explained).tolist()

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_rows_raw": n_rows_raw,
        "n_rows_used": len(df),
        "rows_dropped": {"price_zero": n_price_zero},
        "price_transform": "log1p",
        "feature_columns_used": feature_columns,
        "pca": {
            "n_components": len(feature_columns),
            "explained_variance_ratio": explained,
            "cumulative_explained_variance": cumulative,
            "price_correlation_by_component": [
                {"component": f"PC{i + 1}", "correlation_with_price": c} for i, c in enumerate(price_corr)
            ],
            "correlation_threshold": PRICE_CORR_THRESHOLD,
            "price_correlated_components": [f"PC{i + 1}" for i in selected],
        },
        "feature_ranking": feature_ranking,
        "borough_concentration": borough_concentration,
    }

    (OUTPUT_DIR / "analysis_results.json").write_text(json.dumps(results, indent=2))

    cleaned_columns = [
        "id",
        "name",
        "host_id",
        "host_name",
        "host_since",
        "neighbourhood",
        "neighbourhood_group",
        "latitude",
        "longitude",
        "room_type",
        "accommodates",
        "bathrooms_text",
        "bathrooms",
        "bathrooms_imputed",
        "bedrooms",
        "bedrooms_imputed",
        "beds",
        "beds_imputed",
        "price_clean",
        "price_log",
        "number_of_reviews",
        "last_review",
    ]
    df[cleaned_columns].to_csv(OUTPUT_DIR / "cleaned_listings.csv", index=False)

    pca_scores_df = pd.DataFrame(scores, columns=[f"PC{i + 1}" for i in range(scores.shape[1])])
    pca_scores_df.insert(0, "id", df["id"].to_numpy())
    pca_scores_df.to_csv(OUTPUT_DIR / "pca_scores.csv", index=False)

    print(
        f"loaded {n_rows_raw} rows, dropped {n_price_zero} zero-price rows, "
        f"wrote output/cleaned_listings.csv, output/pca_scores.csv, output/analysis_results.json"
    )


if __name__ == "__main__":
    main()
