# streamlit_app.py
# ========================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO

# Set page config
st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")
st.title("🏠 Ames Housing Price Predictor")
st.markdown("Upload a `test.csv` file to get price predictions and download the results.")

# Load trained model
MODEL_PATH = "best_model.pkl"
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Required feature list (same as training)
features = [
    "Gr Liv Area", "Overall Qual", "Total Bsmt SF", "Garage Cars", "Garage Area",
    "Year Built", "Exter Qual", "Kitchen Qual", "Neighborhood",
    "Age", "Bathrooms", "TotalSF"
]

# Upload CSV
uploaded_file = st.file_uploader("Upload test.csv", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        # Feature Engineering
        df["Age"] = df["Yr Sold"] - df["Year Built"]
        df["Bathrooms"] = df["Full Bath"] + 0.5 * df["Half Bath"]
        df["TotalSF"] = df["Total Bsmt SF"] + df["1st Flr SF"] + df["2nd Flr SF"]

        # Extract required features
        X = df[features].copy()
        X = X.fillna(X.mode().iloc[0])
        X = X.fillna(X.median(numeric_only=True))

        # Predict
        pred_log = model.predict(X)
        pred_price = np.expm1(pred_log) * 1e5

        # Prepare result DataFrame
        df_results = pd.DataFrame()
        if "Id" in df.columns:
            df_results["Id"] = df["Id"]
        df_results["Predicted Price (₹)"] = pred_price.round(0)

        # Display results
        st.subheader("📊 Predictions")
        st.dataframe(df_results.head())

        # Download button
        csv_data = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Predictions as CSV",
            data=csv_data,
            file_name="predicted_prices.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")