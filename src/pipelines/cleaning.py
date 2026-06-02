import pandas as pd
import numpy as np
import os

from src.utils.cleaning_utils import (
    CATEGORICAL_TYPO_CORRECTIONS,
    apply_categorical_corrections,
    normalize_binary_columns,
    normalize_and_flag_missing,
    drop_missing_critical_rows,
    drop_missing_target_rows,
    parse_timestamp,
    detect_datatype_anomalies,
    fix_numeric_columns,
    handle_excluded_missingness,
    report_missing_columns,
    build_model_df,
    impute_numeric_columns,
    impute_binary_columns,
    impute_categorical_columns
)


CRITICAL_COLUMNS = ["transaction_id", "timestamp"]

TARGET_COLUMN = "is_fraud"

IMPUTATION_EXCLUDE_COLS = [
    "transaction_id",
    "customer_id",
    "device_id",
    "ip_address",
    "timestamp",
    "is_fraud"
]

EXCLUDED_FILLABLE_COLS = [
    "customer_id",
    "device_id",
    "ip_address"
]

NUMERIC_COLUMNS = [
    "amount_src",
    "amount_usd",
    "fee",
    "exchange_rate_src_to_dest",
    "ip_risk_score",
    "account_age_days",
    "device_trust_score",
    "chargeback_history_count",
    "risk_score_internal",
    "txn_velocity_1h",
    "txn_velocity_24h",
    "corridor_risk"
]

BINARY_COLUMNS = [
    "new_device",
    "location_mismatch"
]

CATEGORICAL_COLUMNS = [
    "home_country",
    "source_currency",
    "dest_currency",
    "channel",
    "ip_country",
    "kyc_tier"
]


def run_cleaning(input_file, output_file):

    print("Starting data cleaning...\n")

    print("Loading ingestion output dataset...")
    df = pd.read_csv(input_file)
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    # ---------------------------------------------------------
    # Handle missingness in non-imputable identifier columns
    # ---------------------------------------------------------

    print("\nHandling missingness in non-imputable identifier columns...")

    df, missingness_indicators = handle_excluded_missingness(
        df, EXCLUDED_FILLABLE_COLS, sentinel="missing"
    )

    if missingness_indicators:
        print(f"Indicators added: {missingness_indicators}")
    else:
        print("No indicator columns added.")

    binary_columns_extended = BINARY_COLUMNS + missingness_indicators

    # ---------------------------------------------------------
    # Normalize binary columns
    # ---------------------------------------------------------

    print("\nNormalizing binary columns to {0, 1, NaN}...")
    df = normalize_binary_columns(df, binary_columns_extended)
    print("Binary normalization complete.")

    # ---------------------------------------------------------
    # Normalize categoricals and flag placeholders as NaN
    # ---------------------------------------------------------

    print("\nNormalizing categorical values and flagging placeholders as missing...")
    df = normalize_and_flag_missing(df)
    print("Normalization complete.")

    # ---------------------------------------------------------
    # Apply known categorical typo corrections.
    # Runs AFTER normalize_and_flag_missing so the lookup
    # dict (which is keyed in lowercase) matches cleanly.
    # ---------------------------------------------------------

    print("\nApplying categorical typo corrections...")

    df, typo_changes = apply_categorical_corrections(
        df, CATEGORICAL_TYPO_CORRECTIONS
    )

    if typo_changes:
        print("Typo corrections applied:")
        for col, n in typo_changes.items():
            print(f"  {col}: {n} cells corrected")
    else:
        print("No typo corrections needed.")

    # ---------------------------------------------------------
    # Drop rows with missing critical identifiers
    # ---------------------------------------------------------

    print("\nChecking critical identifier columns...")
    df, critical_rows_removed = drop_missing_critical_rows(df, CRITICAL_COLUMNS)
    print(f"Rows removed due to missing identifiers: {critical_rows_removed}")

    # ---------------------------------------------------------
    # Parse timestamp
    # ---------------------------------------------------------

    print("\nParsing timestamp column...")
    df, parse_failures = parse_timestamp(df)
    print(f"Timestamp parse failures removed: {parse_failures}")

    # ---------------------------------------------------------
    # Drop missing-target rows
    # ---------------------------------------------------------

    print(f"\nChecking target column ({TARGET_COLUMN}) for missing values...")
    df, target_rows_removed = drop_missing_target_rows(df, TARGET_COLUMN)
    print(f"Rows removed due to missing target: {target_rows_removed}")

    # ---------------------------------------------------------
    # Separate target before imputation
    # ---------------------------------------------------------

    if TARGET_COLUMN in df.columns:
        target_series = df[TARGET_COLUMN].copy()
        df = df.drop(columns=[TARGET_COLUMN])
        target_present = True
    else:
        target_series = None
        target_present = False

    # ---------------------------------------------------------
    # Restrict to defined column sets
    # ---------------------------------------------------------

    numeric_cols     = [c for c in NUMERIC_COLUMNS         if c in df.columns]
    binary_cols      = [c for c in binary_columns_extended if c in df.columns]
    categorical_cols = [c for c in CATEGORICAL_COLUMNS     if c in df.columns]

    print(f"\nNumeric columns defined:     {len(numeric_cols)}")
    print(f"Binary columns defined:      {len(binary_cols)}  "
          f"(includes {len(missingness_indicators)} missingness indicator(s))")
    print(f"Categorical columns defined: {len(categorical_cols)}")

    # ---------------------------------------------------------
    # Datatype anomalies and numeric fixing
    # ---------------------------------------------------------

    print("\nChecking numeric columns for unexpected string values...")
    datatype_issues = detect_datatype_anomalies(df, numeric_cols)

    if datatype_issues:
        print("Datatype issues detected:")
        for col, count in datatype_issues.items():
            print(f"  {col}: {count} suspicious entries")
    else:
        print("No datatype inconsistencies detected.")

    print("\nFixing string-formatted numeric columns...")
    df, fixed_numeric_cols = fix_numeric_columns(df, numeric_cols)
    if fixed_numeric_cols:
        print(f"Columns fixed: {fixed_numeric_cols}")
    else:
        print("No numeric columns required string fixing.")

    # ---------------------------------------------------------
    # Pre-imputation missingness snapshot
    # ---------------------------------------------------------

    imputable_cols = numeric_cols + binary_cols + categorical_cols
    pre_impute_missing = report_missing_columns(df, subset=imputable_cols)

    print("\nMissing-value snapshot BEFORE imputation (imputable columns only):")

    if pre_impute_missing:
        total = sum(pre_impute_missing.values())
        for col, count in pre_impute_missing.items():
            kind = (
                "numeric"     if col in numeric_cols
                else "binary"      if col in binary_cols
                else "categorical"
            )
            print(f"  {col:<28} {count:>6}  ({kind})")
        print(f"  {'TOTAL':<28} {total:>6}")
    else:
        print("  No missing values in imputable columns.")

    # ---------------------------------------------------------
    # Build model_df and impute
    # ---------------------------------------------------------

    print("\nPreparing encoded dataset for ML imputation...")
    model_df, label_encoders = build_model_df(
        df, categorical_cols, exclude_cols=IMPUTATION_EXCLUDE_COLS
    )
    print("Encoding complete.")

    print("\nRunning ML-based numeric imputation...")
    df, model_df, numeric_imputed = impute_numeric_columns(
        df, model_df, numeric_cols
    )
    print(f"Numeric columns imputed: {numeric_imputed}" if numeric_imputed
          else "No numeric columns required imputation.")

    print("\nRunning ML-based binary imputation...")
    df, model_df, binary_imputed = impute_binary_columns(
        df, model_df, binary_cols
    )
    print(f"Binary columns imputed: {binary_imputed}" if binary_imputed
          else "No binary columns required imputation.")

    print("\nRunning ML-based categorical imputation...")
    df, categorical_imputed = impute_categorical_columns(
        df, model_df, categorical_cols, label_encoders
    )
    print(f"Categorical columns imputed: {categorical_imputed}" if categorical_imputed
          else "No categorical columns required imputation.")

    # ---------------------------------------------------------
    # Re-attach target
    # ---------------------------------------------------------

    if target_present:
        df[TARGET_COLUMN] = target_series

    # ---------------------------------------------------------
    # Final missingness audit
    # ---------------------------------------------------------

    all_missing_by_col = report_missing_columns(df)
    remaining_missing  = int(df.isna().sum().sum())

    imputable_missing_by_col = report_missing_columns(df, subset=imputable_cols)
    imputable_missing_total  = sum(imputable_missing_by_col.values())

    print(f"\nRemaining missing values (entire dataset): {remaining_missing}")
    if all_missing_by_col:
        print("Missing values by column (entire dataset):")
        for col, count in all_missing_by_col.items():
            tag = "imputable" if col in imputable_cols else "excluded/target"
            print(f"  {col:<28} {count:>6}  ({tag})")
    else:
        print("No missing values remain in any column.")

    print(f"\nRemaining missing values (imputable columns only): {imputable_missing_total}")
    print("All imputable columns are complete." if imputable_missing_total == 0
          else "Some imputable columns still contain missing values — review above.")

    # ---------------------------------------------------------
    # Save and report
    # ---------------------------------------------------------

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

    print("\n==============================")
    print("CLEANING SUMMARY")
    print("==============================")
    print(f"Final rows:    {len(df)}")
    print(f"Final columns: {len(df.columns)}")
    print(f"Rows removed (missing identifiers): {critical_rows_removed}")
    print(f"Rows removed (timestamp failures):  {parse_failures}")
    print(f"Rows removed (missing target):      {target_rows_removed}")
    print(f"Typo corrections applied:    {typo_changes}")
    print(f"Missingness indicators added: {missingness_indicators}")
    print(f"Numeric columns fixed:       {len(fixed_numeric_cols)}")
    print(f"Numeric columns imputed:     {len(numeric_imputed)}")
    print(f"Binary columns imputed:      {len(binary_imputed)}")
    print(f"Categorical columns imputed: {len(categorical_imputed)}")
    print(f"Remaining missing (all cols):       {remaining_missing}")
    print(f"Remaining missing (imputable cols): {imputable_missing_total}")
    print(f"\nDataset saved to:\n{output_file}")
    print("\nCleaning completed successfully.")

    return {
        "status": "success",
        "file": output_file,
        "rows": len(df),
        "columns": len(df.columns),
        "critical_rows_removed": critical_rows_removed,
        "timestamp_failures": parse_failures,
        "target_rows_removed": target_rows_removed,
        "typo_corrections": typo_changes,
        "missingness_indicators": missingness_indicators,
        "numeric_cols_fixed": fixed_numeric_cols,
        "numeric_imputed": numeric_imputed,
        "binary_imputed": binary_imputed,
        "categorical_imputed": categorical_imputed,
        "pre_impute_missing": pre_impute_missing,
        "remaining_missing_all": remaining_missing,
        "remaining_missing_imputable": imputable_missing_total,
        "remaining_missing_by_col": all_missing_by_col
    }