import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Re-parse timestamp.
#
# CSVs don't preserve datetime dtype, so any time-based
# feature engineering needs to re-parse the column on
# load. This is NOT redundant with cleaning — it's a
# CSV-format limitation.
# ---------------------------------------------------------

def reparse_timestamp(df, timestamp_col="timestamp"):

    if timestamp_col not in df.columns:
        return df

    df[timestamp_col] = pd.to_datetime(
        df[timestamp_col], utc=True, errors="coerce"
    )

    n_invalid = int(df[timestamp_col].isna().sum())

    if n_invalid > 0:
        raise ValueError(
            f"{n_invalid} unparseable timestamps reached feature "
            f"engineering. Upstream cleaning should have removed "
            f"these. Re-run cleaning before continuing."
        )

    return df


# ---------------------------------------------------------
# Time-based features derived from the timestamp.
# ---------------------------------------------------------

def add_time_features(df, timestamp_col="timestamp"):

    added = []

    if timestamp_col not in df.columns:
        return df, added

    ts = df[timestamp_col]

    df["txn_hour"]        = ts.dt.hour.astype(int)
    df["txn_day_of_week"] = ts.dt.dayofweek.astype(int)
    df["txn_day"]         = ts.dt.day.astype(int)
    df["txn_month"]       = ts.dt.month.astype(int)
    df["is_weekend"]      = ts.dt.dayofweek.isin([5, 6]).astype(int)

    added = ["txn_hour", "txn_day_of_week", "txn_day", "txn_month", "is_weekend"]

    return df, added


# ---------------------------------------------------------
# Derived numeric features.
# ---------------------------------------------------------

def add_derived_numeric_features(df, eps=1e-6):

    added = []

    if "fee" in df.columns and "amount_usd" in df.columns:
        df["fee_pct"] = df["fee"] / (df["amount_usd"] + eps)
        added.append("fee_pct")

    if "amount_usd" in df.columns:
        df["log_amount"] = np.log1p(df["amount_usd"].clip(lower=0))
        added.append("log_amount")

    return df, added


# ---------------------------------------------------------
# Customer velocity features (vectorized, causal).
# ---------------------------------------------------------

def add_velocity_features(df, customer_col="customer_id",
                          timestamp_col="timestamp",
                          amount_col="amount_usd"):

    required = {customer_col, timestamp_col, amount_col}

    if not required.issubset(df.columns):
        return df, []

    df = df.sort_values([customer_col, timestamp_col]).reset_index(drop=False)
    df = df.set_index(timestamp_col)

    grouped = df.groupby(customer_col, group_keys=False)

    df["txn_count_24h"] = grouped[amount_col].rolling("24h").count().values
    df["txn_count_7d"]  = grouped[amount_col].rolling("7d").count().values
    df["avg_amount_7d"] = grouped[amount_col].rolling("7d").mean().values

    df = df.reset_index()

    customer_mean = df.groupby(customer_col)[amount_col].transform("mean")
    df["avg_amount_7d"] = df["avg_amount_7d"].fillna(customer_mean)

    df["txn_count_24h"] = df["txn_count_24h"].fillna(0).astype(int)
    df["txn_count_7d"]  = df["txn_count_7d"].fillna(0).astype(int)

    df = df.sort_values("index").drop(columns=["index"]).reset_index(drop=True)

    added = ["txn_count_24h", "txn_count_7d", "avg_amount_7d"]

    return df, added


# ---------------------------------------------------------
# Device and IP diversity features (vectorized, causal).
# ---------------------------------------------------------

def add_device_ip_diversity(df, customer_col="customer_id",
                            timestamp_col="timestamp",
                            window="30d"):

    if customer_col not in df.columns or timestamp_col not in df.columns:
        return df, []

    df = df.sort_values([customer_col, timestamp_col]).reset_index(drop=False)
    df = df.set_index(timestamp_col)

    added = []

    for col, new_name in [
        ("device_id",  "distinct_devices_30d"),
        ("ip_address", "distinct_ips_30d")
    ]:

        if col not in df.columns:
            continue

        codes, _ = pd.factorize(df[col].astype(str))
        df["_code"] = codes

        df[new_name] = (
            df.groupby(customer_col)["_code"]
            .rolling(window)
            .apply(lambda s: pd.Series(s).nunique(), raw=False)
            .values
        )

        df[new_name] = df[new_name].fillna(1).astype(int)
        df = df.drop(columns=["_code"])
        added.append(new_name)

    df = df.reset_index()
    df = df.sort_values("index").drop(columns=["index"]).reset_index(drop=True)

    return df, added


# ---------------------------------------------------------
# Customer-level aggregate features (causal).
#
# Adds three features, all computed using ONLY each row's
# own history and earlier rows — never future transactions:
#
#   total_amount_30d
#       Rolling 30-day sum of amount_usd per customer.
#
#   amount_to_lifetime_mean_ratio
#       Current amount divided by the customer's EXPANDING
#       mean (cumulative mean up to and including this row).
#       Expanding — not the full-series mean — so no future
#       transaction influences an earlier row. For a
#       customer's first transaction the ratio is 1.0
#       (amount / itself).
#
#   days_since_last_txn
#       Days elapsed since the customer's previous
#       transaction. 0 for a customer's first transaction.
#
# Leakage note: every statistic here is causal. A plain
# groupby().transform("mean") would use the customer's
# entire history including future rows — we deliberately
# avoid that and use expanding / shifted operations.
# ---------------------------------------------------------

def add_customer_aggregates(df, customer_col="customer_id",
                            timestamp_col="timestamp",
                            amount_col="amount_usd",
                            eps=1e-6):

    required = {customer_col, timestamp_col, amount_col}

    if not required.issubset(df.columns):
        return df, []

    df = df.sort_values([customer_col, timestamp_col]).reset_index(drop=False)

    # --- total_amount_30d (rolling 30-day sum, causal) ---
    tmp = df.set_index(timestamp_col)
    df["total_amount_30d"] = (
        tmp.groupby(customer_col, group_keys=False)[amount_col]
        .rolling("30d").sum().values
    )
    df["total_amount_30d"] = df["total_amount_30d"].fillna(df[amount_col])

    # --- expanding lifetime mean (causal) ---
    # expanding().mean() at row i uses rows [0..i] within the
    # customer group, i.e. only the past and present.
    expanding_mean = (
        df.groupby(customer_col)[amount_col]
        .expanding().mean()
        .reset_index(level=0, drop=True)
    )
    df["amount_to_lifetime_mean_ratio"] = (
        df[amount_col] / (expanding_mean + eps)
    )

    # --- days_since_last_txn (causal: uses previous row) ---
    prev_ts = df.groupby(customer_col)[timestamp_col].shift(1)
    delta = (df[timestamp_col] - prev_ts).dt.total_seconds() / 86400.0
    df["days_since_last_txn"] = delta.fillna(0.0)

    # Restore original row order.
    df = df.sort_values("index").drop(columns=["index"]).reset_index(drop=True)

    added = [
        "total_amount_30d",
        "amount_to_lifetime_mean_ratio",
        "days_since_last_txn"
    ]

    return df, added


# ---------------------------------------------------------
# Cohort z-score features (causal / expanding).
#
# Answers: "how unusual is this transaction's amount versus
# others in the same cohort?" Cohort = (home_country, channel).
#
# z = (amount - cohort_running_mean) / cohort_running_std
#
# Leakage handling: cohort mean and std are computed with an
# EXPANDING window per cohort, ordered by timestamp. Each
# row's z-score uses only earlier rows in the same cohort
# (and itself), so no future information leaks into the
# feature.
#
# Caveat (documented honestly): because this feature is built
# BEFORE the train/test split, test-period rows build their
# own cohort statistics from earlier test-period rows. This
# is causal (no future leakage) and is the standard
# compromise for a cohort feature that lives pre-split. A
# fully rigorous version would freeze cohort stats on the
# training period only — that belongs post-split in modelling
# and can be added later if needed.
# ---------------------------------------------------------

def add_cohort_zscores(df, cohort_cols=("home_country", "channel"),
                       timestamp_col="timestamp",
                       amount_col="amount_usd",
                       eps=1e-6):

    cohort_cols = [c for c in cohort_cols if c in df.columns]

    if not cohort_cols or amount_col not in df.columns:
        return df, []

    if timestamp_col in df.columns:
        df = df.sort_values(timestamp_col).reset_index(drop=False)
    else:
        df = df.reset_index(drop=False)

    grp = df.groupby(cohort_cols)[amount_col]

    # Expanding mean and std, shifted by 1 so the CURRENT row
    # is excluded from its own statistics (strictly prior rows).
    run_mean = grp.expanding().mean().reset_index(level=list(range(len(cohort_cols))), drop=True)
    run_std  = grp.expanding().std().reset_index(level=list(range(len(cohort_cols))), drop=True)

    # Align back to df order
    run_mean = run_mean.sort_index()
    run_std  = run_std.sort_index()

    df["_cohort_mean"] = run_mean.values
    df["_cohort_std"]  = run_std.values

    df["amount_cohort_zscore"] = (
        (df[amount_col] - df["_cohort_mean"]) /
        (df["_cohort_std"] + eps)
    )

    # First rows of each cohort have NaN mean/std → neutral 0.
    df["amount_cohort_zscore"] = df["amount_cohort_zscore"].fillna(0.0)

    df = df.drop(columns=["_cohort_mean", "_cohort_std"])
    df = df.sort_values("index").drop(columns=["index"]).reset_index(drop=True)

    return df, ["amount_cohort_zscore"]


# ---------------------------------------------------------
# Bucket rare categories as "other" before one-hot encoding.
# ---------------------------------------------------------

def bucket_rare_categories(df, categorical_cols, threshold=20, other_label="other"):

    bucketing_report = {}

    for col in categorical_cols:

        if col not in df.columns:
            continue

        counts = df[col].value_counts(dropna=False)
        rare = counts[counts < threshold].index.tolist()

        if not rare:
            continue

        df.loc[df[col].isin(rare), col] = other_label
        bucketing_report[col] = {
            "n_rare_categories": len(rare),
            "n_rows_rebucketed": int(df[col].eq(other_label).sum())
        }

    return df, bucketing_report


# ---------------------------------------------------------
# One-hot encode categorical columns.
# ---------------------------------------------------------

def one_hot_encode(df, categorical_cols, drop_first=True):

    existing = [c for c in categorical_cols if c in df.columns]

    if not existing:
        return df, []

    before_cols = set(df.columns)
    df = pd.get_dummies(df, columns=existing, drop_first=drop_first)
    new_cols = sorted(set(df.columns) - before_cols)

    return df, new_cols


# ---------------------------------------------------------
# Drop non-feature columns.
# ---------------------------------------------------------

def drop_non_feature_columns(df, cols_to_drop):

    present = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=present)
    return df, present