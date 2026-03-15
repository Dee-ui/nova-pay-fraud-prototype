import pandas as pd
import numpy as np
import os

def run_cleaning(input_file, output_file):

    df = pd.read_csv(input_file)

    # Normalize categorical values
    for col in df.select_dtypes(include="object").columns:

        df[col] = df[col].astype(str).str.strip().str.lower()

        df[col] = df[col].replace(
            ["unknown", "na", "n/a", ""],
            np.nan
        )

    # Remove rows missing critical identifiers
    critical_cols = ["transaction_id", "timestamp"]

    for col in critical_cols:
        if col in df.columns:
            df = df[df[col].notna()]

    # Parse timestamp
    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
            utc=True
        )

        df = df[df["timestamp"].notna()]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df.to_csv(output_file, index=False)

    return output_file