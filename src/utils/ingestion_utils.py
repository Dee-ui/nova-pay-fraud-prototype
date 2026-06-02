import pandas as pd


# ---------------------------------------------------------
# Validate dataset schema
# ---------------------------------------------------------

def validate_columns(df, expected_columns):

    actual_columns = df.columns.tolist()

    missing_columns = sorted(
        list(set(expected_columns) - set(actual_columns))
    )

    extra_columns = sorted(
        list(set(actual_columns) - set(expected_columns))
    )

    return {
        "missing_columns": missing_columns,
        "extra_columns": extra_columns
    }


# ---------------------------------------------------------
# Validate timestamp column
# ---------------------------------------------------------

def validate_timestamps(df, timestamp_col="timestamp"):

    invalid_timestamps = 0

    if timestamp_col not in df.columns:

        return df, invalid_timestamps

    parsed_timestamp = pd.to_datetime(
        df[timestamp_col],
        errors="coerce"
    )

    invalid_timestamps = parsed_timestamp.isna().sum()

    df = df.loc[parsed_timestamp.notna()].copy()

    df[timestamp_col] = parsed_timestamp[parsed_timestamp.notna()]

    return df, invalid_timestamps


# ---------------------------------------------------------
# Normalize categorical columns
# ---------------------------------------------------------

def normalize_categorical_columns(df):

    categorical_cols = df.select_dtypes(
        include="object"
    ).columns.tolist()

    for col in categorical_cols:

        df[col] = (
            df[col]
            .str.strip()
            .str.lower()
        )

    return df, categorical_cols


# ---------------------------------------------------------
# Remove duplicate transactions
# ---------------------------------------------------------

def remove_duplicate_transactions(
    df,
    transaction_col="transaction_id"
):

    duplicate_count = 0

    if transaction_col in df.columns:

        duplicate_count = df.duplicated(
            subset=[transaction_col]
        ).sum()

        df = df.drop_duplicates(
            subset=[transaction_col]
        )

    return df, duplicate_count


# ---------------------------------------------------------
# Missing value summary.
# Only returns columns that have at least one missing value,
# sorted by missing percentage descending.
# ---------------------------------------------------------

def get_missing_value_summary(df):

    missing_count = df.isnull().sum()

    missing_percent = (
        missing_count / len(df)
    ) * 100

    summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percent": missing_percent
    })

    summary = summary[summary["missing_count"] > 0]

    return summary.sort_values(
        "missing_percent",
        ascending=False
    )