import pandas as pd
import os

def run_ingestion(input_file, dictionary_file, output_path):

    df = pd.read_csv(input_file, low_memory=False)
    dictionary = pd.read_csv(dictionary_file)

    expected_columns = dictionary["Column Name"].str.strip().tolist()
    actual_columns = df.columns.tolist()

    missing_columns = list(set(expected_columns) - set(actual_columns))

    if len(missing_columns) > 0:
        return {
            "status": "error",
            "missing_columns": missing_columns
        }

    os.makedirs(output_path, exist_ok=True)

    cleaned_file = os.path.join(output_path, "transactions_cleaned.csv")
    df.to_csv(cleaned_file, index=False)

    return {
        "status": "success",
        "file": cleaned_file
    }