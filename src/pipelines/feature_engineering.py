import os
import pandas as pd

from src.utils.feature_engineering_utils import (
    reparse_timestamp,
    add_time_features,
    add_derived_numeric_features,
    add_velocity_features,
    add_device_ip_diversity,
    add_customer_aggregates,
    add_cohort_zscores,
    bucket_rare_categories,
    one_hot_encode,
    drop_non_feature_columns
)


CATEGORICAL_COLUMNS_TO_ENCODE = [
    "channel",
    "kyc_tier",
    "home_country",
    "source_currency",
    "dest_currency",
    "ip_country"
]

RARE_CATEGORY_THRESHOLD = 20

NON_FEATURE_COLUMNS = [
    "transaction_id",
    "customer_id",
    "device_id",
    "ip_address",
    # "timestamp" — KEEP for the time-based split in modelling.
]


def run_feature_engineering(input_file, output_file):

    print("Starting feature engineering...\n")

    print("Loading cleaned dataset...")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    print("\nRe-parsing timestamp...")
    df = reparse_timestamp(df)
    print("Timestamp parsed.")

    print("\nAdding time-based features...")
    df, time_added = add_time_features(df)
    print(f"Added: {time_added}")

    print("\nAdding derived numeric features...")
    df, derived_added = add_derived_numeric_features(df)
    print(f"Added: {derived_added}")

    print("\nAdding customer velocity features...")
    df, velocity_added = add_velocity_features(df)
    print(f"Added: {velocity_added}")

    print("\nAdding device & IP diversity features...")
    df, diversity_added = add_device_ip_diversity(df)
    print(f"Added: {diversity_added}")

    # ---------------------------------------------------------
    # Customer-level aggregates (causal). Needs customer_id and
    # amount_usd, which are still present at this point.
    # ---------------------------------------------------------

    print("\nAdding customer aggregate features...")
    df, aggregates_added = add_customer_aggregates(df)
    print(f"Added: {aggregates_added}")

    # ---------------------------------------------------------
    # Cohort z-scores (causal). Must run BEFORE one-hot encoding
    # because it needs the raw home_country / channel strings.
    # ---------------------------------------------------------

    print("\nAdding cohort z-score features...")
    df, cohort_added = add_cohort_zscores(df)
    print(f"Added: {cohort_added}")

    print("\nBucketing rare categories...")
    df, bucketing_report = bucket_rare_categories(
        df, CATEGORICAL_COLUMNS_TO_ENCODE, threshold=RARE_CATEGORY_THRESHOLD
    )

    if bucketing_report:
        print("Rare-category bucketing:")
        for col, info in bucketing_report.items():
            print(f"  {col}: {info['n_rare_categories']} categories "
                  f"-> 'other' ({info['n_rows_rebucketed']} rows)")
    else:
        print("No rare categories to bucket.")

    print("\nOne-hot encoding categorical columns...")
    df, onehot_cols = one_hot_encode(df, CATEGORICAL_COLUMNS_TO_ENCODE)
    print(f"Created {len(onehot_cols)} one-hot columns.")

    print("\nDropping non-feature columns...")
    df, dropped = drop_non_feature_columns(df, NON_FEATURE_COLUMNS)
    print(f"Dropped: {dropped}")

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    n_missing = int(df.isna().sum().sum())
    print(f"\nFinal shape: {df.shape}")
    print(f"Missing values in final feature matrix: {n_missing}")

    if n_missing > 0:
        missing_cols = df.isna().sum()
        missing_cols = missing_cols[missing_cols > 0].sort_values(ascending=False)
        print("Columns with missing values:")
        for col, count in missing_cols.items():
            print(f"  {col:<32} {count}")
        raise ValueError(
            "Feature matrix contains missing values. Review the new "
            "aggregate / cohort features and rolling windows."
        )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

    print("\n==============================")
    print("FEATURE ENGINEERING SUMMARY")
    print("==============================")
    print(f"Final rows:    {len(df)}")
    print(f"Final columns: {len(df.columns)}")
    print(f"Time features added:      {time_added}")
    print(f"Derived numeric features: {derived_added}")
    print(f"Velocity features:        {velocity_added}")
    print(f"Diversity features:       {diversity_added}")
    print(f"Customer aggregates:      {aggregates_added}")
    print(f"Cohort z-score features:  {cohort_added}")
    print(f"One-hot columns created:  {len(onehot_cols)}")
    print(f"Non-feature cols dropped: {dropped}")
    print(f"\nDataset saved to:\n{output_file}")
    print("\nFeature engineering completed successfully.")

    return {
        "status":             "success",
        "file":               output_file,
        "rows":               len(df),
        "columns":            len(df.columns),
        "time_features":      time_added,
        "derived_features":   derived_added,
        "velocity_features":  velocity_added,
        "diversity_features": diversity_added,
        "customer_aggregates": aggregates_added,
        "cohort_features":    cohort_added,
        "bucketing_report":   bucketing_report,
        "onehot_columns":     onehot_cols,
        "dropped_columns":    dropped
    }