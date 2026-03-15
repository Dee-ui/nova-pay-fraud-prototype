import pandas as pd
import numpy as np
import joblib
import os
import re

# metadata columns (not model features)
META_COLUMNS = [
    "transaction_id",
    "customer_id",
    "device_id",
    "ip_address",
    "timestamp"
]


def predict_transactions(feature_file, model_file):

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")

    df = pd.read_csv(feature_file, low_memory=False)

    # --------------------------------------------------
    # Store metadata
    # --------------------------------------------------

    metadata = df[[c for c in META_COLUMNS if c in df.columns]].copy()

    # --------------------------------------------------
    # Remove metadata + target
    # --------------------------------------------------

    drop_cols = META_COLUMNS.copy()

    if "is_fraud" in df.columns:
        drop_cols.append("is_fraud")

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # --------------------------------------------------
    # Fix datatypes (safe numeric conversion)
    # --------------------------------------------------

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    # --------------------------------------------------
    # Detect categoricals
    # --------------------------------------------------

    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # --------------------------------------------------
    # One hot encoding
    # --------------------------------------------------

    if len(categorical_cols) > 0:

        df = pd.get_dummies(
            df,
            columns=categorical_cols,
            drop_first=False
        )

    # --------------------------------------------------
    # Clean column names (same as notebook)
    # --------------------------------------------------

    df.columns = [
        re.sub('[^A-Za-z0-9_]+', '_', col)
        for col in df.columns
    ]

    # --------------------------------------------------
    # Load trained model
    # --------------------------------------------------

    model = joblib.load(model_file)

    # --------------------------------------------------
    # Get model feature names (works for CatBoost & sklearn)
    # --------------------------------------------------

    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)

    elif hasattr(model, "feature_names_"):
        model_features = list(model.feature_names_)

    else:
        raise ValueError("Model does not contain stored feature names")

    # --------------------------------------------------
    # Align features
    # --------------------------------------------------

    missing_features = [f for f in model_features if f not in df.columns]

    for col in missing_features:
        df[col] = 0

    df = df[model_features]

    # --------------------------------------------------
    # Run prediction
    # --------------------------------------------------

    fraud_prob = model.predict_proba(df)[:, 1]

    fraud_pred = (fraud_prob >= 0.5).astype(int)

    # --------------------------------------------------
    # Combine predictions with metadata
    # --------------------------------------------------

    results = metadata.copy()

    results["fraud_probability"] = fraud_prob
    results["fraud_prediction"] = fraud_pred

    return results