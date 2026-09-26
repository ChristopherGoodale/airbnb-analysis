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


st.set_page_config(page_title="NYC Airbnb Price Analysis", layout="wide")

results = load_results()
cleaned = load_cleaned()

st.title("NYC Airbnb Price Analysis")
st.caption(f"Generated {results['generated_at']} from {results['n_rows_used']} listings. Run `python analyze.py` to refresh.")

tab1, tab2, tab3 = st.tabs(["Price Factors (PCA)", "Borough Concentration & Map", "Data Explorer"])

with tab1:
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

with tab2:
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

with tab3:
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
