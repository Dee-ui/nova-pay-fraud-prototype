import pandas as pd
import os

from src.utils.ingestion_utils import (
    validate_columns,
    validate_timestamps,
    normalize_categorical_columns,
    remove_duplicate_transactions,
    get_missing_value_summary
)


# ---------------------------------------------------------
# Critical columns that must exist for the pipeline to run
# ---------------------------------------------------------

CRITICAL_COLUMNS = ["transaction_id", "timestamp"]


def run_ingestion(
    input_file,
    dictionary_file,
    output_path
):

    print("Starting data ingestion...\n")

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    print("Loading transaction dataset...")

    df = pd.read_csv(
        input_file,
        low_memory=False
    )

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # ---------------------------------------------------------
    # Load dictionary
    # ---------------------------------------------------------

    print("\nLoading data dictionary...")

    dictionary = pd.read_csv(dictionary_file)

    expected_columns = (
        dictionary["Column Name"]
        .str.strip()
        .tolist()
    )

    # ---------------------------------------------------------
    # Validate schema
    # ---------------------------------------------------------

    print("\nValidating dataset schema...")

    validation_results = validate_columns(
        df,
        expected_columns
    )

    missing_columns = validation_results["missing_columns"]
    extra_columns = validation_results["extra_columns"]

    if missing_columns:

        print("\nMissing columns detected:")

        for col in missing_columns:
            print(f" - {col}")

    else:
        print("No missing columns detected.")

    if extra_columns:

        print("\nExtra columns detected:")

        for col in extra_columns:
            print(f" - {col}")

    else:
        print("No unexpected columns detected.")

    # ---------------------------------------------------------
    # Hard stop if critical columns are missing
    # ---------------------------------------------------------

    missing_critical = [
        col for col in CRITICAL_COLUMNS
        if col in missing_columns
    ]

    if missing_critical:
        raise ValueError(
            f"Critical columns missing from dataset. "
            f"Pipeline cannot continue. "
            f"Missing: {missing_critical}"
        )

    # ---------------------------------------------------------
    # Validate timestamps
    # ---------------------------------------------------------

    print("\nValidating timestamps...")

    df, invalid_timestamps = validate_timestamps(df)

    print(
        f"Invalid timestamps removed: "
        f"{invalid_timestamps}"
    )

    # ---------------------------------------------------------
    # Normalize categoricals
    # ---------------------------------------------------------

    print("\nNormalizing categorical columns...")

    df, categorical_cols = normalize_categorical_columns(df)

    print(
        f"Categorical columns normalized: "
        f"{len(categorical_cols)}"
    )

    # ---------------------------------------------------------
    # Remove duplicate transactions
    # ---------------------------------------------------------

    print("\nRemoving duplicate transactions...")

    df, duplicate_count = remove_duplicate_transactions(df)

    print(
        f"Duplicate transactions removed: "
        f"{duplicate_count}"
    )

    # ---------------------------------------------------------
    # Basic missing value summary
    # ---------------------------------------------------------

    print("\nGenerating missing value summary...")

    missing_summary = get_missing_value_summary(df)

    print(
        missing_summary.head(10)
    )

    # ---------------------------------------------------------
    # Save cleaned ingestion dataset
    # ---------------------------------------------------------

    os.makedirs(output_path, exist_ok=True)

    output_file = os.path.join(
        output_path,
        "transactions_ingested.csv"
    )

    df.to_csv(output_file, index=False)

    # ---------------------------------------------------------
    # Final ingestion report
    # ---------------------------------------------------------

    print("\n==============================")
    print("INGESTION SUMMARY")
    print("==============================")

    print(f"Final rows: {len(df)}")
    print(f"Final columns: {len(df.columns)}")

    print(
        f"Duplicate transactions removed: "
        f"{duplicate_count}"
    )

    print(
        f"Invalid timestamps removed: "
        f"{invalid_timestamps}"
    )

    print(f"\nDataset saved to:")
    print(output_file)

    print("\nIngestion completed successfully.")

    return {
        "status": "success",
        "file": output_file,
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_count": duplicate_count,
        "invalid_timestamps": invalid_timestamps,
        "missing_columns": missing_columns,
        "extra_columns": extra_columns
    }