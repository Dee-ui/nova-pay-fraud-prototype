import streamlit as st
import pandas as pd
import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.ingestion import run_ingestion
from src.cleaning import run_cleaning
from src.feature_engineering import run_feature_engineering
from src.prediction import predict_transactions

from config.config import (
    DATA_DICTIONARY_FILE,
    PROCESSED_DATA_PATH,
    PREPARED_DATA_FILE,
    FEATURE_DATA_FILE,
    BEST_MODEL_FILE
)

st.title("NovaPay Fraud Detection System")

st.write("Upload a transaction dataset to detect fraud.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    temp_path = os.path.join(raw_dir, uploaded_file.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("File uploaded successfully")

    # -----------------------------
    # Ingestion
    # -----------------------------

    st.write("Running ingestion...")

    result = run_ingestion(
        temp_path,
        DATA_DICTIONARY_FILE,
        PROCESSED_DATA_PATH
    )

    if result["status"] == "error":

        st.error("Column mismatch detected")

        st.write("Missing columns:")
        st.write(result["missing_columns"])

        st.stop()

    st.success("Ingestion completed")

    # -----------------------------
    # Cleaning
    # -----------------------------

    st.write("Running cleaning...")

    cleaned_file = run_cleaning(
        result["file"],
        PREPARED_DATA_FILE
    )

    st.success("Cleaning completed")

    # -----------------------------
    # Feature Engineering
    # -----------------------------

    st.write("Running feature engineering...")

    feature_file = run_feature_engineering(
        cleaned_file,
        FEATURE_DATA_FILE
    )

    st.success("Feature engineering completed")

    # -----------------------------
    # Prediction
    # -----------------------------

    st.write("Running fraud prediction...")

    predictions = predict_transactions(
        feature_file,
        BEST_MODEL_FILE
    )

    st.success("Prediction completed")

    st.subheader("Prediction Results")

    st.dataframe(predictions.head())

    fraud_count = predictions["fraud_prediction"].sum()

    total = len(predictions)

    fraud_rate = fraud_count / total

    st.metric("Total Transactions", total)
    st.metric("Predicted Fraud Transactions", fraud_count)
    st.metric("Fraud Rate", f"{fraud_rate:.2%}")

    st.subheader("Fraud Distribution")

    st.bar_chart(predictions["fraud_prediction"].value_counts())