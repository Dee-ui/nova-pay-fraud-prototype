import os
import pandas as pd

from src.utils.prediction_utils import (
    load_model_bundle,
    prepare_features_for_prediction,
    run_predictions
)


TARGET_COLUMN = "is_fraud"

META_COLUMNS = [
    "transaction_id",
    "customer_id",
    "device_id",
    "ip_address",
    "timestamp"
]


def predict_transactions(feature_file, model_dir, model_name, output_file=None):

    print("Starting prediction...\n")

    # ---------------------------------------------------------
    # 1. Load the model bundle (model + scaler + metadata +
    #    anomaly detectors)
    # ---------------------------------------------------------

    print("Loading model bundle...")
    bundle = load_model_bundle(model_dir, model_name)
    print(f"  Model: {model_name}")
    print(f"  Threshold: {bundle['threshold']:.4f}")
    print(f"  Expects {len(bundle['feature_columns'])} features")

    # ---------------------------------------------------------
    # 2. Load incoming feature-engineered data
    # ---------------------------------------------------------

    print("\nLoading feature-engineered data...")
    df = pd.read_csv(feature_file, low_memory=False)
    print(f"  Rows: {df.shape[0]}")

    # Preserve any metadata columns for the output (if present).
    metadata_present = [c for c in META_COLUMNS if c in df.columns]
    metadata = df[metadata_present].copy() if metadata_present else None

    # ---------------------------------------------------------
    # 3. Prepare the feature matrix exactly as the model expects
    # ---------------------------------------------------------

    print("\nPreparing features (anomaly scores, alignment, scaling)...")
    X_model = prepare_features_for_prediction(df, bundle, TARGET_COLUMN, META_COLUMNS)

    # ---------------------------------------------------------
    # 4. Predict
    # ---------------------------------------------------------

    print("\nRunning predictions...")
    results = run_predictions(bundle, X_model, metadata)
    print(f"  Predicted {len(results)} rows.  "
          f"Flagged fraud: {int(results['fraud_prediction'].sum())}")

    # ---------------------------------------------------------
    # 5. Save (optional)
    # ---------------------------------------------------------

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        results.to_csv(output_file, index=False)
        print(f"\nPredictions saved to:\n{output_file}")

    print("\nPrediction completed successfully.")

    return results