import os
import re
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sklearn.model_selection import (
    train_test_split,
    TimeSeriesSplit,
    cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    precision_recall_curve
)
from sklearn.exceptions import ConvergenceWarning

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    IsolationForest
)
from sklearn.neural_network import MLPClassifier

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# HBOS (Histogram-Based Outlier Score) from PyOD
from pyod.models.hbos import HBOS


optuna.logging.set_verbosity(optuna.logging.WARNING)


# =========================================================
# Data loading and dtype validation
# =========================================================

def load_feature_matrix(input_file, target_col):

    df = pd.read_csv(input_file, low_memory=False)

    object_cols = df.drop(columns=[target_col], errors="ignore").select_dtypes(
        include=["object"]
    ).columns.tolist()
    object_cols = [c for c in object_cols if c != "timestamp"]

    if object_cols:
        raise ValueError(
            f"Modeling received non-numeric columns from feature "
            f"engineering: {object_cols}. Feature engineering "
            f"should have one-hot encoded all categoricals."
        )

    return df


def sanitize_feature_names(df):
    df.columns = [
        re.sub(r"[^A-Za-z0-9_]+", "_", str(col))
        for col in df.columns
    ]
    return df


# =========================================================
# Train / val / test split (time-based when possible)
# =========================================================

def split_train_val_test(
    df, target_col, timestamp_col="timestamp", random_state=42
):

    if timestamp_col in df.columns:

        print(f"Using TIME-BASED split on `{timestamp_col}`")

        df = df.copy()
        df[timestamp_col] = pd.to_datetime(
            df[timestamp_col], utc=True, errors="coerce"
        )

        if df[timestamp_col].isna().any():
            raise ValueError("Timestamp column contains unparseable values.")

        df = df.sort_values(timestamp_col).reset_index(drop=True)

        n = len(df)
        train_end = int(n * 0.70)
        val_end   = int(n * 0.85)

        train = df.iloc[:train_end]
        val   = df.iloc[train_end:val_end]
        test  = df.iloc[val_end:]
        meta_cols = [timestamp_col]

    else:
        print("WARNING: timestamp missing — falling back to random split.")
        train, temp = train_test_split(
            df, test_size=0.30, random_state=random_state,
            stratify=df[target_col]
        )
        val, test = train_test_split(
            temp, test_size=0.50, random_state=random_state,
            stratify=temp[target_col]
        )
        meta_cols = []

    def _xy(part):
        X = part.drop(columns=[target_col] + meta_cols, errors="ignore")
        y = part[target_col]
        return X, y

    X_train, y_train = _xy(train)
    X_val,   y_val   = _xy(val)
    X_test,  y_test  = _xy(test)

    X_train, X_val  = X_train.align(X_val,  join="left", axis=1, fill_value=0)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    return X_train, X_val, X_test, y_train, y_val, y_test


# =========================================================
# Unsupervised anomaly-score features.
#
# Fits two anomaly detectors on the TRAINING features only,
# then scores train/val/test. Adds two columns:
#   iso_forest_score : IsolationForest score (higher = more anomalous)
#   hbos_score       : HBOS outlier score   (higher = more anomalous)
#
# Leakage safety: both detectors are fit ONLY on X_train.
#
# HBOS robustness: HBOS builds per-feature histograms and
# fails ("bins must be monotonically increasing") on constant
# or near-constant columns — common with one-hot and
# *_was_missing indicators inside a single split. We therefore
# fit HBOS only on the non-constant subset of columns, and
# guard the whole HBOS step so a failure degrades gracefully
# to a neutral score instead of crashing the pipeline.
#
# Returns the three augmented frames plus the fitted detectors
# AND the column subset HBOS was fit on (so serving can
# reproduce the exact same scoring).
# ---------------------------------------------------------

def add_anomaly_features(X_train, X_val, X_test, random_state=42):

    # --- IsolationForest: tolerant of constant columns ---
    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1
    )
    iso.fit(X_train)

    # --- Determine the non-constant columns for HBOS ---
    # A column with zero variance on TRAIN breaks HBOS's
    # histogram binning. Drop those columns for HBOS only.
    stds = X_train.std(axis=0, numeric_only=True)
    hbos_cols = stds[stds > 0].index.tolist()

    hbos = None
    if hbos_cols:
        try:
            hbos = HBOS()
            hbos.fit(X_train[hbos_cols].values)
        except Exception as e:
            print(f"  WARNING: HBOS fit failed ({e}); "
                  f"hbos_score will be set to 0.")
            hbos = None

    def _augment(X):
        X = X.copy()
        X["iso_forest_score"] = -iso.score_samples(X)

        if hbos is not None:
            try:
                # Align to the exact columns HBOS was fit on.
                X_hbos = X.reindex(columns=hbos_cols, fill_value=0).values
                X["hbos_score"] = hbos.decision_function(X_hbos)
            except Exception as e:
                print(f"  WARNING: HBOS scoring failed ({e}); "
                      f"hbos_score set to 0.")
                X["hbos_score"] = 0.0
        else:
            X["hbos_score"] = 0.0

        return X

    X_train_aug = _augment(X_train)
    X_val_aug   = _augment(X_val)
    X_test_aug  = _augment(X_test)

    detectors = {
        "isolation_forest": iso,
        "hbos":             hbos,
        "hbos_cols":        hbos_cols
    }

    return X_train_aug, X_val_aug, X_test_aug, detectors


# =========================================================
# Preprocessing
# =========================================================

def fit_preprocessing(X_train):

    vt = VarianceThreshold(threshold=0.0)
    X_train_vt = vt.fit_transform(X_train)

    kept_mask = vt.get_support()
    kept_cols = X_train.columns[kept_mask].tolist()
    dropped   = X_train.columns[~kept_mask].tolist()

    scaler = StandardScaler()
    scaler.fit(X_train_vt)

    return {
        "variance_filter": vt,
        "scaler":          scaler,
        "kept_cols":       kept_cols,
        "dropped_cols":    dropped
    }


def apply_preprocessing(X, preproc, scale=True):
    X_vt = preproc["variance_filter"].transform(X)
    if scale:
        return preproc["scaler"].transform(X_vt)
    return X_vt


# =========================================================
# Metrics
# =========================================================

def compute_metrics(y_true, y_pred, y_proba):
    return {
        "roc_auc":   float(roc_auc_score(y_true, y_proba)),
        "pr_auc":    float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "brier":     float(brier_score_loss(y_true, y_proba))
    }


def evaluate_at_default_threshold(model, X_eval, y_eval):
    proba = model.predict_proba(X_eval)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    return compute_metrics(y_eval, pred, proba), proba


# =========================================================
# Threshold optimization
# =========================================================

def find_best_threshold(y_true, y_proba, target="f1"):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    precisions = precisions[:-1]
    recalls    = recalls[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        f1s = 2 * (precisions * recalls) / (precisions + recalls)
    f1s = np.nan_to_num(f1s)
    best_idx = int(np.argmax(f1s))

    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precisions[best_idx]),
        "recall":    float(recalls[best_idx]),
        "f1":        float(f1s[best_idx])
    }


# =========================================================
# Candidate families
# =========================================================

def build_linear_candidates(random_state):
    return {
        "LogReg_L2": (
            LogisticRegression(
                l1_ratio=0.0, C=1.0, max_iter=10000, solver="saga",
                class_weight="balanced", random_state=random_state
            ),
            {"scale": True}
        ),
        "LogReg_L1": (
            LogisticRegression(
                l1_ratio=1.0, C=1.0, max_iter=10000, solver="saga",
                class_weight="balanced", random_state=random_state
            ),
            {"scale": True}
        ),
        "LogReg_ElasticNet": (
            LogisticRegression(
                l1_ratio=0.5, C=1.0, max_iter=10000, solver="saga",
                class_weight="balanced", random_state=random_state
            ),
            {"scale": True}
        )
    }


def build_tree_candidates(random_state, scale_pos_weight):
    return {
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=300, max_depth=None, min_samples_leaf=5,
                class_weight="balanced", random_state=random_state, n_jobs=-1
            ),
            {"scale": False}
        ),
        "XGBoost": (
            xgb.XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                eval_metric="logloss", scale_pos_weight=scale_pos_weight,
                random_state=random_state, tree_method="hist", n_jobs=-1
            ),
            {"scale": False}
        ),
        "LightGBM": (
            lgb.LGBMClassifier(
                n_estimators=400, num_leaves=63, max_depth=-1,
                learning_rate=0.05, min_child_samples=20,
                class_weight="balanced", random_state=random_state,
                n_jobs=-1, verbose=-1
            ),
            {"scale": False}
        ),
        "CatBoost": (
            CatBoostClassifier(
                iterations=400, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
                auto_class_weights="Balanced", random_state=random_state, verbose=0
            ),
            {"scale": False}
        ),
        "HistGB": (
            HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.05, max_depth=None,
                min_samples_leaf=20, class_weight="balanced",
                random_state=random_state
            ),
            {"scale": False}
        )
    }


def build_neural_candidates(random_state):
    return {
        "MLP_wide_1L": (
            MLPClassifier(
                hidden_layer_sizes=(128,), activation="relu", solver="adam",
                alpha=1e-4, learning_rate="adaptive", max_iter=300,
                random_state=random_state, early_stopping=True,
                n_iter_no_change=15, validation_fraction=0.15
            ),
            {"scale": True}
        ),
        "MLP_tanh_2L": (
            MLPClassifier(
                hidden_layer_sizes=(64, 32), activation="tanh", solver="adam",
                alpha=1e-3, learning_rate="adaptive", max_iter=300,
                random_state=random_state, early_stopping=True,
                n_iter_no_change=15, validation_fraction=0.15
            ),
            {"scale": True}
        ),
        "MLP_deep_3L": (
            MLPClassifier(
                hidden_layer_sizes=(64, 32, 16), activation="relu", solver="adam",
                alpha=1e-3, learning_rate="adaptive", max_iter=400,
                random_state=random_state, early_stopping=True,
                n_iter_no_change=15, validation_fraction=0.15
            ),
            {"scale": True}
        )
    }


# =========================================================
# Candidate selection on a sample
# =========================================================

def train_and_score_candidates(
    candidates, X_train_full, y_train_full,
    X_val_raw, X_val_scaled, y_val,
    sample_frac=0.30, random_state=42
):

    sample_idx = X_train_full.sample(
        frac=sample_frac, random_state=random_state
    ).index

    X_sample = X_train_full.loc[sample_idx]
    y_sample = y_train_full.loc[sample_idx]

    results = []

    for name, (model, opts) in candidates.items():

        scale = opts["scale"]
        print(f"  → Training {name} (scaled={scale})...")

        if scale:
            X_sample_used = StandardScaler().fit_transform(X_sample)
            X_val_used    = X_val_scaled
        else:
            X_sample_used = X_sample
            X_val_used    = X_val_raw

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning,
                                    message=".*X has feature names.*")
            model.fit(X_sample_used, y_sample)

        metrics, _ = evaluate_at_default_threshold(model, X_val_used, y_val)
        metrics["candidate"] = name
        metrics["scaled"]    = scale
        results.append(metrics)

        print(f"    PR-AUC={metrics['pr_auc']:.4f}  "
              f"ROC-AUC={metrics['roc_auc']:.4f}  "
              f"F1={metrics['f1']:.4f}")

    results_df = pd.DataFrame(results).sort_values(
        "pr_auc", ascending=False
    ).reset_index(drop=True)

    return results_df


# =========================================================
# Optuna search spaces (unchanged)
# =========================================================

def _suggest_logreg(trial, l1_ratio, random_state):
    return LogisticRegression(
        C=trial.suggest_float("C", 1e-3, 100.0, log=True),
        l1_ratio=l1_ratio, max_iter=10000, solver="saga",
        class_weight=trial.suggest_categorical("class_weight", ["balanced", None]),
        random_state=random_state
    )


def _suggest_xgboost(trial, scale_pos_weight, random_state):
    return xgb.XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
        eval_metric="logloss", scale_pos_weight=scale_pos_weight,
        random_state=random_state, tree_method="hist", n_jobs=-1
    )


def _suggest_lightgbm(trial, random_state):
    return lgb.LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
        num_leaves=trial.suggest_int("num_leaves", 15, 255),
        max_depth=trial.suggest_int("max_depth", -1, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        class_weight="balanced", random_state=random_state, n_jobs=-1, verbose=-1
    )


def _suggest_catboost(trial, random_state):
    return CatBoostClassifier(
        iterations=trial.suggest_int("iterations", 200, 1000, step=100),
        depth=trial.suggest_int("depth", 4, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        border_count=trial.suggest_int("border_count", 32, 254),
        auto_class_weights="Balanced", random_state=random_state, verbose=0
    )


def _suggest_random_forest(trial, random_state):
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
        max_depth=trial.suggest_int("max_depth", 5, 40),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        class_weight="balanced", random_state=random_state, n_jobs=-1
    )


def _suggest_histgb(trial, random_state):
    return HistGradientBoostingClassifier(
        max_iter=trial.suggest_int("max_iter", 200, 1000, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 100),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        class_weight="balanced", random_state=random_state
    )


def _suggest_mlp(trial, hidden_template, activation, random_state):
    return MLPClassifier(
        hidden_layer_sizes=hidden_template, activation=activation, solver="adam",
        alpha=trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
        learning_rate_init=trial.suggest_float("learning_rate_init", 1e-5, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        learning_rate="adaptive", max_iter=400, random_state=random_state,
        early_stopping=True, n_iter_no_change=15, validation_fraction=0.15
    )


def get_suggester(name, scale_pos_weight, random_state):

    if name in ("LogReg_L2", "LogReg_L1", "LogReg_ElasticNet"):
        l1_ratio = {"LogReg_L2": 0.0, "LogReg_L1": 1.0, "LogReg_ElasticNet": 0.5}[name]
        return lambda trial: _suggest_logreg(trial, l1_ratio, random_state)
    if name == "XGBoost":
        return lambda trial: _suggest_xgboost(trial, scale_pos_weight, random_state)
    if name == "LightGBM":
        return lambda trial: _suggest_lightgbm(trial, random_state)
    if name == "CatBoost":
        return lambda trial: _suggest_catboost(trial, random_state)
    if name == "RandomForest":
        return lambda trial: _suggest_random_forest(trial, random_state)
    if name == "HistGB":
        return lambda trial: _suggest_histgb(trial, random_state)
    if name == "MLP_wide_1L":
        return lambda trial: _suggest_mlp(trial, (128,), "relu", random_state)
    if name == "MLP_tanh_2L":
        return lambda trial: _suggest_mlp(trial, (64, 32), "tanh", random_state)
    if name == "MLP_deep_3L":
        return lambda trial: _suggest_mlp(trial, (64, 32, 16), "relu", random_state)

    raise ValueError(f"No Optuna suggester defined for {name}")


# =========================================================
# Optuna tuning with time-aware CV
# =========================================================

def tune_finalist_optuna(
    name, scale, X_train, y_train, scaler_full,
    scale_pos_weight, use_time_cv=True, n_trials=30, random_state=42
):

    suggester = get_suggester(name, scale_pos_weight, random_state)
    cv = TimeSeriesSplit(n_splits=3) if use_time_cv else 3

    X_train_used = scaler_full.transform(X_train) if scale else X_train.values
    y_train_arr = y_train.values

    def objective(trial):
        model = suggester(trial)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            scores = cross_val_score(
                model, X_train_used, y_train_arr,
                cv=cv, scoring="average_precision", n_jobs=-1
            )
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state),
        pruner=MedianPruner(n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  Best params for {name}: {study.best_params}")
    print(f"  Best CV PR-AUC for {name}: {study.best_value:.4f}")

    best_model = suggester(study.best_trial)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        best_model.fit(X_train_used, y_train_arr)

    return best_model, study.best_params, study.best_value


# =========================================================
# Save model bundle.
#
# Now also optionally persists the fitted anomaly detectors
# so the serving pipeline can reproduce the iso_forest_score
# and hbos_score features at inference time.
# =========================================================

def save_model_bundle(
    output_dir, name, model, scaler, feature_cols,
    threshold, metrics, params, scale_required,
    anomaly_detectors=None
):

    os.makedirs(output_dir, exist_ok=True)

    model_path    = os.path.join(output_dir, f"{name}.pkl")
    scaler_path   = os.path.join(output_dir, f"{name}_scaler.pkl")
    meta_path     = os.path.join(output_dir, f"{name}_metadata.json")
    detectors_path = os.path.join(output_dir, "anomaly_detectors.pkl")

    joblib.dump(model, model_path)
    if scale_required:
        joblib.dump(scaler, scaler_path)
    if anomaly_detectors is not None:
        joblib.dump(anomaly_detectors, detectors_path)

    metadata = {
        "model_name":         name,
        "scale_required":     bool(scale_required),
        "decision_threshold": float(threshold),
        "feature_columns":    list(feature_cols),
        "best_params":        params if params else {},
        "test_metrics":       metrics,
        "anomaly_detectors_file": "anomaly_detectors.pkl" if anomaly_detectors else None
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return {
        "model_path":  model_path,
        "scaler_path": scaler_path if scale_required else None,
        "meta_path":   meta_path
    }


# =========================================================
# PR curve
# =========================================================

def plot_pr_curve(y_true, y_proba, output_dir, name):

    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    baseline = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recalls, precisions, label=name)
    ax.axhline(baseline, color="red", linestyle="--",
               label=f"baseline ({baseline:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall curve — {name}")
    ax.legend()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"pr_curve_{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path