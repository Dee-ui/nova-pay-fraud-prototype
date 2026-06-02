import os
import json
import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Load everything needed to make predictions: the model,
# its metadata (feature columns, threshold, scale flag),
# the scaler (if needed), and the anomaly detectors.
# ---------------------------------------------------------

def load_model_bundle(model_dir, model_name):

    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    meta_path  = os.path.join(model_dir, f"{model_name}_metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    model = joblib.load(model_path)

    with open(meta_path) as f:
        metadata = json.load(f)

    scaler = None
    if metadata.get("scale_required"):
        scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
        scaler = joblib.load(scaler_path)

    anomaly_detectors = None
    detectors_file = metadata.get("anomaly_detectors_file")
    if detectors_file:
        det_path = os.path.join(model_dir, detectors_file)
        if os.path.exists(det_path):
            anomaly_detectors = joblib.load(det_path)

    return {
        "model":             model,
        "scaler":            scaler,
        "metadata":          metadata,
        "feature_columns":   metadata["feature_columns"],
        "threshold":         metadata["decision_threshold"],
        "scale_required":    metadata["scale_required"],
        "anomaly_detectors": anomaly_detectors
    }


# ---------------------------------------------------------
# Prepare the incoming feature matrix to match EXACTLY what
# the model was trained on:
#   1. Drop metadata + target columns.
#   2. Re-apply the saved anomaly detectors (iso_forest_score,
#      hbos_score) — the SAME fitted detectors from training,
#      so no re-fitting and no skew.
#   3. Reindex to the model's expected feature columns
#      (adds any missing as 0, drops any extras, fixes order).
#   4. Scale if the model requires it (using the saved scaler).
#
# This replaces the old approach of re-encoding from scratch,
# which could produce mismatched columns and silent skew.
# ---------------------------------------------------------

def prepare_features_for_prediction(df, bundle, target_col, meta_columns):

    df = df.copy()

    # 1. Drop metadata and target if present.
    drop_cols = [c for c in meta_columns if c in df.columns]
    if target_col in df.columns:
        drop_cols.append(target_col)
    X = df.drop(columns=drop_cols, errors="ignore")

    # Sanitize names the same way modelling did.
    import re
    X.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", str(c)) for c in X.columns]

    # 2. Re-apply anomaly detectors if available.
    detectors = bundle["anomaly_detectors"]
    if detectors is not None:
        iso       = detectors["isolation_forest"]
        hbos      = detectors.get("hbos")
        hbos_cols = detectors.get("hbos_cols", [])

        # IsolationForest: align to the columns it was fit on.
        iso_cols = list(getattr(iso, "feature_names_in_", X.columns))
        X_for_iso = X.reindex(columns=iso_cols, fill_value=0)
        X["iso_forest_score"] = -iso.score_samples(X_for_iso)

        # HBOS: align to its non-constant column subset, with fallback.
        if hbos is not None and hbos_cols:
            X_for_hbos = X.reindex(columns=hbos_cols, fill_value=0).values
            X["hbos_score"] = hbos.decision_function(X_for_hbos)
        else:
            X["hbos_score"] = 0.0

    # 3. Align to the model's expected feature columns.
    feature_cols = bundle["feature_columns"]
    X = X.reindex(columns=feature_cols, fill_value=0)

    # 4. Scale if required.
    if bundle["scale_required"] and bundle["scaler"] is not None:
        X_arr = bundle["scaler"].transform(X)
        X = pd.DataFrame(X_arr, columns=feature_cols, index=X.index)

    return X


# ---------------------------------------------------------
# Run predictions and assemble the output frame.
# Applies the SAVED decision threshold (not a hardcoded 0.5).
# ---------------------------------------------------------

def run_predictions(bundle, X_model, metadata=None):

    model = bundle["model"]
    threshold = bundle["threshold"]

    proba = model.predict_proba(X_model)[:, 1]
    pred  = (proba >= threshold).astype(int)

    if metadata is not None:
        results = metadata.reset_index(drop=True).copy()
    else:
        results = pd.DataFrame(index=range(len(proba)))

    results["fraud_probability"] = proba
    results["fraud_prediction"]  = pred

    return results