import streamlit as st
import pandas as pd
import os

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

st.title("Fraud Detection System")

st.write("Upload a transaction dataset to detect fraud.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    temp_path = os.path.join("data/raw", uploaded_file.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("File uploaded successfully")

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

    st.write("Running cleaning...")

    cleaned_file = run_cleaning(
        result["file"],
        PREPARED_DATA_FILE
    )

    st.success("Cleaning completed")

    st.write("Running feature engineering...")

    feature_file = run_feature_engineering(
        cleaned_file,
        FEATURE_DATA_FILE
    )

    st.success("Feature engineering completed")

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