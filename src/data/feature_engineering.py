import pandas as pd

def run_feature_engineering(input_file, output_file):

    df = pd.read_csv(input_file)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["txn_hour"] = df["timestamp"].dt.hour

    df.to_csv(output_file, index=False)

    return output_file