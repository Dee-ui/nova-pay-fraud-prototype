import os
import pandas as pd

from src.pipelines.cleaning import (
    NUMERIC_COLUMNS,
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    EXCLUDED_FILLABLE_COLS,
    TARGET_COLUMN
)

from src.utils.eda_utils import (
    verify_imputation,
    schema_snapshot,
    class_balance,
    numeric_statistics,
    outlier_report,
    tail_extremes_report,
    plot_numeric_distributions,
    categorical_fraud_rates,
    binary_fraud_rates,
    time_patterns,
    correlation_analysis,
    cardinality_report,
    build_summary_payload
)


ID_COLUMNS_FOR_CARDINALITY = [
    "customer_id",
    "device_id",
    "ip_address"
]


def run_eda(input_file, output_dir):

    print("Starting EDA...\n")

    print("Loading cleaned dataset...")
    df = pd.read_csv(input_file)
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Resolve which configured columns actually exist
    # ---------------------------------------------------------

    numeric_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]

    binary_cols = [c for c in BINARY_COLUMNS if c in df.columns] + [
        f"{c}_was_missing" for c in EXCLUDED_FILLABLE_COLS
        if f"{c}_was_missing" in df.columns
    ]

    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    id_cols          = [c for c in ID_COLUMNS_FOR_CARDINALITY if c in df.columns]

    print(f"\nNumeric columns:     {len(numeric_cols)}")
    print(f"Binary columns:      {len(binary_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"ID columns:          {len(id_cols)}")

    # ---------------------------------------------------------
    # 0. Verify imputation — fail loudly if anything's missing
    #
    # Sentinel-filled ID columns (customer_id, device_id,
    # ip_address) must also be complete and are included in
    # the check.
    # ---------------------------------------------------------

    print("\nVerifying imputation completeness...")

    expected_complete = (
        numeric_cols + binary_cols + categorical_cols + id_cols
    )

    imputation_check = verify_imputation(
        df, expected_complete, TARGET_COLUMN, output_dir
    )

    if not imputation_check["passed"]:
        print("Imputation verification FAILED:")
        print(f"  Features with missing values: {imputation_check['feature_cols_with_missing']}")
        print(f"  Target missing count:         {imputation_check['target_missing_count']}")
        raise ValueError(
            "Cleaned dataset still contains missing values in feature or "
            "target columns. Fix the cleaning step before running EDA."
        )

    print("Imputation verification passed — no missing values in features or target.")

    # ---------------------------------------------------------
    # 1. Schema snapshot
    # ---------------------------------------------------------

    print("\nGenerating schema snapshot...")
    snapshot = schema_snapshot(df)
    snapshot.to_csv(os.path.join(output_dir, "schema_snapshot.csv"))

    # ---------------------------------------------------------
    # 2. Class balance (pie + bar)
    # ---------------------------------------------------------

    print("Computing class balance...")
    class_summary, fraud_rate = class_balance(df, TARGET_COLUMN, output_dir)
    print(f"Fraud rate: {fraud_rate:.4f}")

    # ---------------------------------------------------------
    # 3. Numeric stats, outliers, tail extremes, plots
    # ---------------------------------------------------------

    print("Computing numeric statistics...")
    numeric_stats = numeric_statistics(df, numeric_cols, TARGET_COLUMN, output_dir)

    print("Computing outlier report (descriptive)...")
    outliers = outlier_report(df, numeric_cols, output_dir)

    print("Computing tail extremes report...")
    extremes = tail_extremes_report(df, numeric_cols, output_dir)

    print("Plotting numeric distributions...")
    plot_numeric_distributions(df, numeric_cols, TARGET_COLUMN, output_dir)

    # ---------------------------------------------------------
    # 4. Categorical fraud rates + count plots
    # ---------------------------------------------------------

    print("Analyzing categorical fraud rates...")
    categorical_summaries = categorical_fraud_rates(
        df, categorical_cols, TARGET_COLUMN, output_dir
    )

    # ---------------------------------------------------------
    # 5. Binary fraud rates
    # ---------------------------------------------------------

    print("Analyzing binary feature fraud rates...")
    binary_summary = binary_fraud_rates(df, binary_cols, TARGET_COLUMN, output_dir)

    # ---------------------------------------------------------
    # 6. Time patterns (not persisted to df)
    # ---------------------------------------------------------

    print("Analyzing time-based patterns...")
    time_summary = time_patterns(df, TARGET_COLUMN, output_dir)

    # ---------------------------------------------------------
    # 7. Correlation analysis
    # ---------------------------------------------------------

    print("Computing correlation analysis...")
    corr_matrix, target_corr, high_corr_pairs = correlation_analysis(
        df, numeric_cols, TARGET_COLUMN, output_dir
    )

    # ---------------------------------------------------------
    # 8. Cardinality of identifier columns
    # ---------------------------------------------------------

    print("Computing cardinality report...")
    cardinality = cardinality_report(df, id_cols, output_dir)

    # ---------------------------------------------------------
    # 9. Dashboard JSON summary
    # ---------------------------------------------------------

    print("Writing dashboard summary payload...")
    summary_payload, summary_path = build_summary_payload(
        df, TARGET_COLUMN, fraud_rate, target_corr,
        high_corr_pairs, imputation_check, output_dir
    )

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    print("\n==============================")
    print("EDA SUMMARY")
    print("==============================")
    print(f"Rows analyzed:        {len(df)}")
    print(f"Fraud count:          {int(df[TARGET_COLUMN].sum())}")
    print(f"Fraud rate:           {fraud_rate:.4f}")
    print(f"Numeric cols:         {len(numeric_cols)}")
    print(f"Binary cols:          {len(binary_cols)}")
    print(f"Categorical cols:     {len(categorical_cols)}")
    print(f"High-corr pairs:      {len(high_corr_pairs)}")
    print(f"\nArtifacts written to: {output_dir}")
    print(f"Dashboard summary:    {summary_path}")
    print("\nEDA completed successfully.")

    return {
        "status":           "success",
        "rows":             len(df),
        "columns":          df.shape[1],
        "fraud_rate":       fraud_rate,
        "output_dir":       output_dir,
        "summary_payload":  summary_payload,
        "numeric_cols":     numeric_cols,
        "binary_cols":      binary_cols,
        "categorical_cols": categorical_cols
    }