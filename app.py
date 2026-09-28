"""Streamlit viewer for the findings in output/. Run analyze.py first.

    streamlit run app.py
"""
import json

import pandas as pd
import streamlit as st

OUTPUT_DIR = "output"


@st.cache_data
def load_cleaned() -> pd.DataFrame:
    return pd.read_csv(f"{OUTPUT_DIR}/cleaned_listings.csv")


@st.cache_data
def load_pca_scores() -> pd.DataFrame:
    return pd.read_csv(f"{OUTPUT_DIR}/pca_scores.csv")


@st.cache_data
def load_results() -> dict:
    with open(f"{OUTPUT_DIR}/analysis_results.json") as f:
        return json.load(f)


@st.cache_data
def load_regression_results() -> dict:
    with open(f"{OUTPUT_DIR}/regression_results.json") as f:
        return json.load(f)


@st.cache_data
def load_regression_coefficients() -> pd.DataFrame:
    return pd.read_csv(f"{OUTPUT_DIR}/regression_coefficients.csv")


st.set_page_config(page_title="NYC Airbnb Price Analysis", layout="wide")

results = load_results()
regression_results = load_regression_results()
cleaned = load_cleaned()

st.title("NYC Airbnb Price Analysis")
st.caption(f"Generated {results['generated_at']} from {results['n_rows_used']} listings. Run `python analyze.py` to refresh.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Pitch", "Price Factors (PCA)", "Price Factors (Regression)", "Borough Concentration & Map", "Data Explorer"]
)

with tab1:
    st.subheader("Pitch")
    st.write("Content coming soon.")

with tab2:
    st.subheader("Ranked factors affecting price")
    ranking = pd.DataFrame(results["feature_ranking"]).set_index("feature")["importance_score"]
    st.bar_chart(ranking)

    st.subheader("Variance explained per component")
    pca_info = results["pca"]
    variance_df = pd.DataFrame(
        {
            "explained_variance_ratio": pca_info["explained_variance_ratio"],
            "cumulative_explained_variance": pca_info["cumulative_explained_variance"],
        },
        index=[f"PC{i + 1}" for i in range(pca_info["n_components"])],
    )
    st.line_chart(variance_df)

    price_corr = pca_info["price_correlated_components"]
    corr_values = {
        c["component"]: c["correlation_with_price"]
        for c in pca_info["price_correlation_by_component"]
        if c["component"] in price_corr
    }
    st.caption(f"Components correlated with price: {corr_values}")

    pca_top_feature = results["feature_ranking"][0]["feature"]
    regression_top_feature = regression_results["models"]["ols"]["coefficients"][0]["feature"]
    st.info(
        f"PCA's top price-correlated factor is **{pca_top_feature}** (what moves together with price "
        f"in the data's overall variance). The Regression tab ranks **{regression_top_feature}** highest "
        "instead, isolating each feature's effect holding the others constant. Compare the two there."
    )

with tab3:
    st.subheader("Model fit")
    metrics_df = pd.DataFrame(
        [
            {
                "model": name.upper(),
                "test_r2": m["test_r2"],
                "test_rmse": m["test_rmse"],
                "alpha": m.get("alpha"),
                "features_zeroed": m.get("n_features_zeroed", 0),
            }
            for name, m in regression_results["models"].items()
        ]
    ).set_index("model")
    st.dataframe(metrics_df)
    st.caption(
        f"Target: {regression_results['target']} (log price). "
        f"Trained on {regression_results['train_test_split']['n_train']} rows, "
        f"tested on {regression_results['train_test_split']['n_test']} held-out rows. "
        f"{regression_results['note']}"
    )

    st.subheader("Standardized coefficients by model")
    coef_df = load_regression_coefficients()
    coef_df = coef_df.reindex(coef_df["ols_coef"].abs().sort_values(ascending=False).index)
    coef_df = coef_df.set_index("feature").rename(
        columns={"ols_coef": "OLS", "ridge_coef": "Ridge", "lasso_coef": "Lasso"}
    )
    st.bar_chart(coef_df, horizontal=True, sort=False)
    st.caption(
        "Sorted by OLS coefficient magnitude, largest at top. Bars extend right for a positive effect on "
        "price and left for a negative one, holding other features constant."
    )

    ridge_alpha = regression_results["models"]["ridge"]["alpha"]
    lasso_alpha = regression_results["models"]["lasso"]["alpha"]
    lasso_zeroed = regression_results["models"]["lasso"]["n_features_zeroed"]
    st.info(
        f"""
**Model selection and explanation**

- **OLS** minimizes prediction error only. No penalty on coefficient size, so it's the unshrunk baseline the other two are compared against.
- **Ridge** adds a penalty on the *sum of squared* coefficients, shrinking all of them toward zero (harder on features that are redundant with each other). Cross-validation picked alpha = {ridge_alpha:.1f} here.
- **Lasso** adds a penalty on the *sum of absolute* coefficients instead, which can push some all the way to exactly zero. Cross-validation picked alpha = {lasso_alpha:.5f} here, small enough that it zeroed out {lasso_zeroed} of {len(coef_df)} features.

The three bars land close together because cross-validation found that little to no shrinkage gives the best held-out prediction accuracy, meaning these features aren't badly redundant with each other. If they were, Ridge and Lasso would visibly pull away from OLS.

**Why these three, and not something else:** the target (price) is continuous, so a classification model like logistic regression doesn't apply here. All three of these produce a coefficient per feature that's directly readable as "effect on price holding the others constant," which is what this tab is for, unlike a support vector or polynomial model, whose outputs don't hand back a clean per-feature effect size the same way. Running all three together, rather than picking one, is what actually lets you check whether the ranking is stable: if Ridge or Lasso had pulled away from OLS, that would have been the signal that some of these features are redundant with each other and the OLS numbers alone couldn't be trusted.
"""
    )

with tab4:
    st.subheader("Listing share by borough")
    concentration = pd.DataFrame(results["borough_concentration"]).set_index("neighbourhood_group")["pct_of_total"]
    st.bar_chart(concentration)

    st.subheader("Listing map")
    boroughs = cleaned["neighbourhood_group"].unique().tolist()
    selected_boroughs = st.multiselect("Filter by borough", boroughs, default=boroughs)
    map_df = cleaned[cleaned["neighbourhood_group"].isin(selected_boroughs)][["latitude", "longitude"]]
    map_df = map_df.rename(columns={"latitude": "lat", "longitude": "lon"})
    if len(map_df) > 5000:
        map_df = map_df.sample(n=5000, random_state=42)
    st.map(map_df)

with tab5:
    st.subheader("Data explorer")
    col1, col2 = st.columns(2)
    with col1:
        borough_filter = st.multiselect(
            "Borough", cleaned["neighbourhood_group"].unique().tolist(), default=cleaned["neighbourhood_group"].unique().tolist()
        )
    with col2:
        room_type_filter = st.multiselect(
            "Room type", cleaned["room_type"].unique().tolist(), default=cleaned["room_type"].unique().tolist()
        )

    price_min, price_max = float(cleaned["price_clean"].min()), float(cleaned["price_clean"].max())
    price_range = st.slider("Price range ($)", price_min, price_max, (price_min, price_max))

    filtered = cleaned[
        cleaned["neighbourhood_group"].isin(borough_filter)
        & cleaned["room_type"].isin(room_type_filter)
        & cleaned["price_clean"].between(*price_range)
    ]

    display_columns = [
        "id",
        "name",
        "neighbourhood",
        "neighbourhood_group",
        "room_type",
        "accommodates",
        "bathrooms",
        "bedrooms",
        "beds",
        "price_clean",
        "number_of_reviews",
    ]
    st.write(f"{len(filtered)} listings shown")
    st.dataframe(filtered[display_columns])
