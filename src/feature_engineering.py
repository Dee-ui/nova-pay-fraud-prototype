import pandas as pd
import os

def run_feature_engineering(input_file, output_file):

    df = pd.read_csv(input_file)

    # Timestamp features
    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        df["txn_hour"] = df["timestamp"].dt.hour
        df["txn_day_of_week"] = df["timestamp"].dt.dayofweek
        df["txn_month"] = df["timestamp"].dt.month

    # Boolean conversion
    bool_cols = ["new_device", "location_mismatch"]

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df.to_csv(output_file, index=False)

    return output_file