import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier


# ---------------------------------------------------------
# Canonical mappings for binary values.
# Anything not in TRUE_TOKENS or FALSE_TOKENS becomes NaN.
# ---------------------------------------------------------

TRUE_TOKENS  = {"true", "t", "yes", "y", "1", "1.0"}
FALSE_TOKENS = {"false", "f", "no", "n", "0", "0.0"}


# ---------------------------------------------------------
# Canonical typo corrections for known-bad categorical
# values. Keys and values are stored lowercase because
# normalize_and_flag_missing has already lowercased the
# data by the time these corrections run.
#
# This dict is the SINGLE place to add new spelling fixes
# as they're discovered. Each entry is `bad -> good`.
# ---------------------------------------------------------

CATEGORICAL_TYPO_CORRECTIONS = {
    "kyc_tier": {
        "standrd":  "standard",
        "enhancd":  "enhanced"
    },
    "channel": {
        "weeb":     "web",
        "mobi":     "mobile",
        "mobil":    "mobile",
        "mobiile":  "mobile",
        "mobille":  "mobile"
    }
}


# ---------------------------------------------------------
# Apply per-column typo corrections.
#
# Runs AFTER normalize_and_flag_missing so the inputs are
# already lowercased and stripped. Only columns listed in
# `corrections` are touched; everything else passes
# through unchanged.
#
# Returns the corrected df and a dict mapping each column
# to the number of cells changed (for the cleaning report).
# ---------------------------------------------------------

def apply_categorical_corrections(df, corrections):

    changes = {}

    for col, mapping in corrections.items():

        if col not in df.columns:
            continue

        before = df[col].copy()

        df[col] = df[col].replace(mapping)

        n_changed = int((before != df[col]).sum())

        if n_changed > 0:
            changes[col] = n_changed

    return df, changes


# ---------------------------------------------------------
# Normalize binary columns to {0, 1, NaN}.
#
# Runs BEFORE normalize_and_flag_missing so that the
# generic string-lowercase pass doesn't mangle these
# values into "true"/"false" strings.
# ---------------------------------------------------------

def normalize_binary_columns(df, binary_cols):

    for col in binary_cols:

        if col not in df.columns:
            continue

        series = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        mapped = pd.Series(np.nan, index=df.index, dtype=float)

        mapped[series.isin(TRUE_TOKENS)]  = 1.0
        mapped[series.isin(FALSE_TOKENS)] = 0.0

        df[col] = mapped

    return df


# ---------------------------------------------------------
# Normalize categorical values and treat common
# placeholders as missing
# ---------------------------------------------------------

def normalize_and_flag_missing(df):

    NULL_PLACEHOLDERS = ["unknown", "na", "n/a", "none", ""]

    for col in df.select_dtypes(include="object").columns:

        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df[col] = df[col].replace(NULL_PLACEHOLDERS, np.nan)

    return df


# ---------------------------------------------------------
# Drop rows with missing critical identifier columns
# ---------------------------------------------------------

def drop_missing_critical_rows(df, critical_cols):

    rows_before = len(df)

    df = df.dropna(subset=[
        col for col in critical_cols if col in df.columns
    ])

    rows_after = len(df)

    rows_removed = rows_before - rows_after

    return df, rows_removed


# ---------------------------------------------------------
# Drop rows where the target column is missing.
# ---------------------------------------------------------

def drop_missing_target_rows(df, target_col):

    rows_before = len(df)

    if target_col not in df.columns:
        return df, 0

    df = df.dropna(subset=[target_col])

    rows_after = len(df)

    rows_removed = rows_before - rows_after

    return df, rows_removed


# ---------------------------------------------------------
# Parse timestamp column to datetime.
# ---------------------------------------------------------

def parse_timestamp(df, timestamp_col="timestamp"):

    parse_failures = 0

    if timestamp_col not in df.columns:
        return df, parse_failures

    parsed = pd.to_datetime(
        df[timestamp_col],
        errors="coerce",
        utc=True
    )

    parse_failures = parsed.isna().sum()

    df = df.loc[parsed.notna()].copy()

    df[timestamp_col] = parsed[parsed.notna()]

    return df, parse_failures


# ---------------------------------------------------------
# Detect datatype anomalies in numeric columns.
# ---------------------------------------------------------

def detect_datatype_anomalies(df, numeric_cols):

    datatype_issues = {}

    for col in numeric_cols:

        invalid_entries = df[col].apply(
            lambda x: isinstance(x, str)
        ).sum()

        if invalid_entries > 0:
            datatype_issues[col] = invalid_entries

    return datatype_issues


# ---------------------------------------------------------
# Fix numeric columns containing string-formatted numbers.
# ---------------------------------------------------------

def fix_numeric_columns(df, numeric_cols):

    fixed_cols = []

    for col in numeric_cols:

        if col not in df.columns:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
        )

        df[col] = pd.to_numeric(df[col], errors="coerce")

        fixed_cols.append(col)

    return df, fixed_cols


# ---------------------------------------------------------
# Handle missingness in non-imputable identifier columns.
# ---------------------------------------------------------

def handle_excluded_missingness(df, cols, sentinel="missing"):

    indicators_added = []

    for col in cols:

        if col not in df.columns:
            continue

        missing_mask = df[col].isna()

        indicator_col = f"{col}_was_missing"

        df[indicator_col] = missing_mask.astype(float)

        if missing_mask.sum() > 0:
            df[col] = df[col].fillna(sentinel)

        indicators_added.append(indicator_col)

    return df, indicators_added


# ---------------------------------------------------------
# Report which columns contain missing values.
# ---------------------------------------------------------

def report_missing_columns(df, subset=None):

    if subset is not None:
        cols = [c for c in subset if c in df.columns]
        target_df = df[cols]
    else:
        target_df = df

    counts = target_df.isna().sum()

    counts = counts[counts > 0].sort_values(ascending=False)

    return counts.to_dict()


# ---------------------------------------------------------
# Build an encoded copy of the dataframe for ML imputation.
# ---------------------------------------------------------

def build_model_df(df, categorical_cols, exclude_cols):

    model_df = df.copy()

    model_df = model_df.drop(
        columns=[c for c in exclude_cols if c in model_df.columns]
    )

    label_encoders = {}

    for col in categorical_cols:

        if col not in model_df.columns:
            continue

        le = LabelEncoder()

        not_null_mask = model_df[col].notna()

        encoded = pd.Series(np.nan, index=model_df.index, dtype=float)

        encoded[not_null_mask] = le.fit_transform(
            model_df.loc[not_null_mask, col].astype(str)
        )

        model_df[col] = encoded

        label_encoders[col] = le

    return model_df, label_encoders


# ---------------------------------------------------------
# ML-based imputation for numeric columns.
# ---------------------------------------------------------

def impute_numeric_columns(df, model_df, numeric_cols):

    imputed_cols = []

    for col in numeric_cols:

        if col not in model_df.columns:
            continue

        if df[col].isna().sum() == 0:
            continue

        train_mask = model_df[col].notna()
        test_mask  = model_df[col].isna()

        if test_mask.sum() == 0:
            continue

        X_train = model_df.loc[train_mask].drop(columns=[col])
        y_train = model_df.loc[train_mask, col]

        X_test  = model_df.loc[test_mask].drop(columns=[col])

        model = ExtraTreesRegressor(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        df.loc[df[col].isna(), col]             = preds
        model_df.loc[model_df[col].isna(), col] = preds

        imputed_cols.append(col)

    return df, model_df, imputed_cols


# ---------------------------------------------------------
# ML-based imputation for binary columns.
# ---------------------------------------------------------

def impute_binary_columns(df, model_df, binary_cols):

    imputed_cols = []

    for col in binary_cols:

        if col not in model_df.columns:
            continue

        if df[col].isna().sum() == 0:
            continue

        train_mask = model_df[col].notna()
        test_mask  = model_df[col].isna()

        if test_mask.sum() == 0:
            continue

        X_train = model_df.loc[train_mask].drop(columns=[col])
        y_train = model_df.loc[train_mask, col].astype(int)

        X_test  = model_df.loc[test_mask].drop(columns=[col])

        model = ExtraTreesClassifier(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test).astype(int)

        df.loc[df[col].isna(), col]             = preds
        model_df.loc[model_df[col].isna(), col] = preds

        df[col] = df[col].astype(int)

        imputed_cols.append(col)

    return df, model_df, imputed_cols


# ---------------------------------------------------------
# ML-based imputation for categorical columns.
# ---------------------------------------------------------

def impute_categorical_columns(df, model_df, categorical_cols, label_encoders):

    imputed_cols = []

    for col in categorical_cols:

        if col not in model_df.columns:
            continue

        if df[col].isna().sum() == 0:
            continue

        train_mask = model_df[col].notna()
        test_mask  = model_df[col].isna()

        if test_mask.sum() == 0:
            continue

        X_train = model_df.loc[train_mask].drop(columns=[col])
        y_train = model_df.loc[train_mask, col]

        X_test  = model_df.loc[test_mask].drop(columns=[col])

        model = ExtraTreesClassifier(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        decoded = label_encoders[col].inverse_transform(
            preds.astype(int)
        )

        df.loc[df[col].isna(), col] = decoded

        imputed_cols.append(col)

    return df, imputed_cols