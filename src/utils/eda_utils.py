import os
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------

def _save_and_close(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


def _save_table(df, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=True)
    return path


# ---------------------------------------------------------
# Imputation verification.
#
# Confirms the cleaning step produced a fully-imputed
# dataset. Returns a report of any unexpected missingness.
# `expected_complete_cols` is the set of columns that
# MUST have zero NaNs (typically: numeric, binary,
# categorical features and the *_was_missing indicators).
# The target is checked separately and reported but does
# not cause a hard failure here.
# ---------------------------------------------------------

def verify_imputation(df, expected_complete_cols, target_col, output_dir):

    present = [c for c in expected_complete_cols if c in df.columns]

    missing_by_col = (
        df[present]
        .isna().sum()
    )
    missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)

    target_missing = (
        int(df[target_col].isna().sum())
        if target_col in df.columns
        else None
    )

    report = pd.DataFrame({
        "missing_count": missing_by_col
    })

    _save_table(report, output_dir, "imputation_verification.csv")

    payload = {
        "feature_cols_with_missing": missing_by_col.to_dict(),
        "target_missing_count":      target_missing,
        "passed":                    bool(missing_by_col.empty and (target_missing or 0) == 0)
    }

    return payload


# ---------------------------------------------------------
# Schema snapshot
# ---------------------------------------------------------

def schema_snapshot(df):
    snapshot = pd.DataFrame({
        "dtype":          df.dtypes.astype(str),
        "non_null_count": df.notna().sum(),
        "null_count":     df.isna().sum(),
        "null_pct":       (df.isna().mean() * 100).round(2),
        "n_unique":       df.nunique(dropna=True)
    })
    return snapshot.sort_values("null_count", ascending=False)


# ---------------------------------------------------------
# Class balance: pie (dashboard tile) + bar (analytical).
# ---------------------------------------------------------

def class_balance(df, target_col, output_dir):

    counts = df[target_col].value_counts().sort_index()
    rates  = counts / counts.sum()

    summary = pd.DataFrame({
        "count": counts,
        "rate":  rates.round(6)
    })

    # Pie — headline tile
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts, labels=[f"Class {i}" for i in counts.index],
           autopct="%1.2f%%", startangle=90)
    ax.set_title(f"{target_col} distribution (pie)")
    _save_and_close(fig, output_dir, "class_balance_pie.png")

    # Bar — more honest for severe imbalance
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    ax.set_title(f"{target_col} distribution (count)")
    ax.set_xlabel(target_col)
    ax.set_ylabel("count")
    _save_and_close(fig, output_dir, "class_balance_bar.png")

    _save_table(summary, output_dir, "class_balance.csv")

    fraud_rate = float(counts.get(1, 0)) / float(counts.sum())

    return summary, fraud_rate


# ---------------------------------------------------------
# Numeric statistics
# ---------------------------------------------------------

def numeric_statistics(df, numeric_cols, target_col, output_dir):

    cols = [c for c in numeric_cols if c in df.columns]

    overall = df[cols].describe().T
    overall["skew"]     = df[cols].skew()
    overall["kurtosis"] = df[cols].kurtosis()

    by_class = df.groupby(target_col)[cols].mean().T
    by_class.columns = [f"mean_class_{c}" for c in by_class.columns]

    combined = overall.join(by_class)

    _save_table(combined, output_dir, "numeric_statistics.csv")

    return combined


# ---------------------------------------------------------
# Outlier report — IQR-based, descriptive only.
#
# In fraud detection, outliers are often the fraud itself
# and SHOULD NOT be removed based on this report. It exists
# to characterize distribution tails and inform feature
# engineering (e.g., whether to log-transform).
# ---------------------------------------------------------

def outlier_report(df, numeric_cols, output_dir):

    cols = [c for c in numeric_cols if c in df.columns]

    rows = []

    for col in cols:
        s = df[col].dropna()
        if s.empty:
            continue

        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        below = (s < lower).sum()
        above = (s > upper).sum()

        rows.append({
            "column":       col,
            "lower_bound":  lower,
            "upper_bound":  upper,
            "n_below":      int(below),
            "n_above":      int(above),
            "n_outliers":   int(below + above),
            "pct_outliers": round((below + above) / len(s) * 100, 2)
        })

    report = pd.DataFrame(rows).set_index("column").sort_values(
        "pct_outliers", ascending=False
    )

    _save_table(report, output_dir, "outlier_report.csv")

    return report


# ---------------------------------------------------------
# Tail extremes report — top-5 and bottom-5 raw values
# per numeric column. Complements the IQR report by
# showing the actual extreme observations for inspection.
# ---------------------------------------------------------

def tail_extremes_report(df, numeric_cols, output_dir, n=5):

    rows = []

    for col in numeric_cols:
        if col not in df.columns:
            continue
        s = df[col].dropna().sort_values()
        if s.empty:
            continue

        rows.append({
            "column":     col,
            "bottom_n":   json.dumps(s.head(n).tolist()),
            "top_n":      json.dumps(s.tail(n).tolist())
        })

    report = pd.DataFrame(rows).set_index("column")

    _save_table(report, output_dir, "tail_extremes.csv")

    return report


# ---------------------------------------------------------
# Numeric distributions: histogram + boxplot-by-target.
# ---------------------------------------------------------

def plot_numeric_distributions(df, numeric_cols, target_col, output_dir):

    plot_dir = os.path.join(output_dir, "numeric_distributions")
    saved = []

    for col in numeric_cols:
        if col not in df.columns:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[col].dropna(), kde=True, ax=axes[0])
        axes[0].set_title(f"{col} distribution")

        sns.boxplot(x=target_col, y=col, data=df, ax=axes[1])
        axes[1].set_title(f"{col} by {target_col}")

        plt.tight_layout()
        path = _save_and_close(fig, plot_dir, f"{col}.png")
        saved.append(path)

    return saved


# ---------------------------------------------------------
# Categorical fraud rates with baseline reference line.
# Also produces a count plot per column so analysts can
# see how trustworthy each category's fraud rate is.
# ---------------------------------------------------------

def categorical_fraud_rates(df, categorical_cols, target_col, output_dir):

    overall_rate = df[target_col].mean()
    plot_dir = os.path.join(output_dir, "categorical_fraud_rates")
    count_dir = os.path.join(output_dir, "categorical_counts")

    summaries = {}

    for col in categorical_cols:
        if col not in df.columns:
            continue

        pivot = (
            df.groupby(col)[target_col]
            .agg(["count", "mean"])
            .rename(columns={"mean": "fraud_rate"})
        )
        pivot["lift_vs_overall"] = (pivot["fraud_rate"] / overall_rate).round(3)
        pivot = pivot.sort_values("fraud_rate", ascending=False)

        _save_table(pivot, output_dir, f"fraud_rate_by_{col}.csv")

        # Fraud-rate bar with baseline reference line.
        # The red dashed line is the overall fraud rate —
        # bars above it = elevated risk, bars below = safer.
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=pivot.index, y=pivot["fraud_rate"], ax=ax)
        ax.axhline(overall_rate, color="red", linestyle="--",
                   label=f"overall ({overall_rate:.4f})")
        ax.set_title(f"Fraud rate by {col}")
        ax.set_ylabel("Fraud rate")
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        _save_and_close(fig, plot_dir, f"{col}.png")

        # Count plot — how many rows per category. A category
        # with 5 rows and 100% fraud rate is statistical noise;
        # this chart makes that obvious at a glance.
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=pivot.index, y=pivot["count"], ax=ax, color="steelblue")
        ax.set_title(f"Row count by {col}")
        ax.set_ylabel("count")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        _save_and_close(fig, count_dir, f"{col}.png")

        summaries[col] = pivot

    return summaries


# ---------------------------------------------------------
# Binary fraud rates (including *_was_missing indicators).
# ---------------------------------------------------------

def binary_fraud_rates(df, binary_cols, target_col, output_dir):

    overall_rate = df[target_col].mean()
    rows = []

    for col in binary_cols:
        if col not in df.columns:
            continue

        grouped = df.groupby(col)[target_col].agg(["count", "mean"])

        for value in grouped.index:
            rows.append({
                "column":          col,
                "value":           int(value),
                "count":           int(grouped.loc[value, "count"]),
                "fraud_rate":      round(float(grouped.loc[value, "mean"]), 6),
                "lift_vs_overall": round(float(grouped.loc[value, "mean"]) / overall_rate, 3)
            })

    report = pd.DataFrame(rows)
    _save_table(report, output_dir, "fraud_rate_by_binary.csv")
    return report


# ---------------------------------------------------------
# Time-based patterns — derived locally, NOT persisted.
# The cleaned dataset is NOT modified by this function.
# Only aggregated CSVs and plots are saved.
# ---------------------------------------------------------

def time_patterns(df, target_col, output_dir, timestamp_col="timestamp"):

    if timestamp_col not in df.columns:
        return None

    ts = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
    hour = ts.dt.hour
    dow  = ts.dt.dayofweek

    hourly = df.groupby(hour)[target_col].agg(["count", "mean"])
    hourly.columns = ["count", "fraud_rate"]
    hourly.index.name = "hour_of_day"

    daily = df.groupby(dow)[target_col].agg(["count", "mean"])
    daily.columns = ["count", "fraud_rate"]
    daily.index.name = "day_of_week"

    _save_table(hourly, output_dir, "fraud_rate_by_hour.csv")
    _save_table(daily,  output_dir, "fraud_rate_by_day_of_week.csv")

    overall_rate = df[target_col].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(x=hourly.index, y=hourly["fraud_rate"], ax=ax, marker="o")
    ax.axhline(overall_rate, color="red", linestyle="--",
               label=f"overall ({overall_rate:.4f})")
    ax.set_title("Fraud rate by hour of day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Fraud rate")
    ax.legend()
    _save_and_close(fig, output_dir, "fraud_rate_by_hour.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=daily.index, y=daily["fraud_rate"], ax=ax)
    ax.axhline(overall_rate, color="red", linestyle="--",
               label=f"overall ({overall_rate:.4f})")
    ax.set_title("Fraud rate by day of week (0=Mon)")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Fraud rate")
    ax.legend()
    _save_and_close(fig, output_dir, "fraud_rate_by_day_of_week.png")

    return {"hourly": hourly, "daily": daily}


# ---------------------------------------------------------
# Correlation analysis.
#
# Pearson correlation measures LINEAR relationships only.
# Low correlation with a binary target does not mean low
# predictive value — non-linear models (trees, boosting)
# can still find combinatorial patterns. High correlation
# between features signals multicollinearity, which mainly
# affects linear-model interpretability, not tree-based
# model accuracy.
# ---------------------------------------------------------

def correlation_analysis(df, numeric_cols, target_col, output_dir):

    cols = [c for c in numeric_cols if c in df.columns]
    cols_with_target = cols + [target_col] if target_col in df.columns else cols

    corr = df[cols_with_target].corr(method="pearson")

    _save_table(corr, output_dir, "correlation_matrix.csv")

    target_corr = (
        corr[target_col]
        .drop(target_col)
        .sort_values(key=abs, ascending=False)
        .to_frame(name="corr_with_target")
    )
    _save_table(target_corr, output_dir, "correlation_with_target.csv")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, cbar_kws={"shrink": 0.75})
    ax.set_title("Pearson correlation matrix")
    _save_and_close(fig, output_dir, "correlation_heatmap.png")

    # Also flag highly-correlated feature pairs (|r| > 0.85)
    # — these are multicollinearity candidates.
    feature_corr = corr.drop(index=target_col, columns=target_col, errors="ignore")
    pairs = []
    cols_only = feature_corr.columns.tolist()
    for i in range(len(cols_only)):
        for j in range(i + 1, len(cols_only)):
            r = feature_corr.iloc[i, j]
            if abs(r) > 0.85:
                pairs.append({
                    "feature_a": cols_only[i],
                    "feature_b": cols_only[j],
                    "pearson_r": round(float(r), 4)
                })
    high_corr_pairs = pd.DataFrame(pairs)
    _save_table(high_corr_pairs, output_dir, "high_correlation_pairs.csv")

    return corr, target_corr, high_corr_pairs


# ---------------------------------------------------------
# Cardinality report
# ---------------------------------------------------------

def cardinality_report(df, cols, output_dir, top_n=10):

    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        n_unique = df[col].nunique(dropna=True)
        top_values = df[col].value_counts(dropna=False).head(top_n).to_dict()
        rows.append({
            "column":     col,
            "n_unique":   int(n_unique),
            "top_values": json.dumps({str(k): int(v) for k, v in top_values.items()})
        })

    report = pd.DataFrame(rows).set_index("column")
    _save_table(report, output_dir, "cardinality_report.csv")
    return report


# ---------------------------------------------------------
# Dashboard summary payload — JSON-friendly.
# Includes context notes the dashboard can render next to
# the relevant charts (e.g. interpreting low correlation).
# ---------------------------------------------------------

def build_summary_payload(
    df,
    target_col,
    fraud_rate,
    target_corr,
    high_corr_pairs,
    imputation_check,
    output_dir
):

    max_abs_target_corr = (
        float(target_corr["corr_with_target"].abs().max())
        if not target_corr.empty
        else None
    )

    payload = {
        "rows":            int(len(df)),
        "columns":         int(df.shape[1]),
        "fraud_count":     int(df[target_col].sum()),
        "fraud_rate":      round(float(fraud_rate), 6),
        "imputation_check": imputation_check,
        "top_features_by_abs_corr": (
            target_corr["corr_with_target"].head(10).round(4).to_dict()
        ),
        "high_correlation_pairs": high_corr_pairs.to_dict(orient="records"),
        "max_abs_target_corr":    round(max_abs_target_corr, 4) if max_abs_target_corr else None,
        "notes": {
            "correlation": (
                "Pearson correlation captures LINEAR relationships. "
                "Low values with a binary target are expected and do not "
                "imply weak models — tree-based methods can still capture "
                "non-linear and combinatorial signal."
            ),
            "outliers": (
                "Outliers reported via IQR are descriptive only. In fraud "
                "detection, extreme values are often the fraud itself and "
                "should not be removed."
            ),
            "baseline_line": (
                "Red dashed lines on per-category charts mark the overall "
                "fraud rate. Bars above the line indicate categories with "
                "elevated risk; bars below indicate safer-than-average."
            )
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "eda_summary.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    return payload, path