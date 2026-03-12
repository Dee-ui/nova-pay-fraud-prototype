import pandas as pd

def run_cleaning(input_file, output_file):

    df = pd.read_csv(input_file)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["transaction_id", "timestamp"])

    df.to_csv(output_file, index=False)

    return output_file