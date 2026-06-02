import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# Reuse the exact split + anomaly logic from modelling so the
# evaluation data matches what the model was trained/tested on.
from src.utils.modelling_utils import (
    load_feature_matrix,
    sanitize_feature_names,
    split_train_val_test,
    add_anomaly_features
)


def _save_table(df, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    df.to_csv(path)
    return path


def _save_fig(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------
# Load the winning model + its metadata from the model dir,
# using the metrics CSV to identify which model won.
# ---------------------------------------------------------

def load_winning_model(model_dir, metrics_file):

    metrics = pd.read_csv(metrics_file)
    winner_row = metrics[metrics["is_winner"]].iloc[0]
    model_name = winner_row["model"]

    model = joblib.load(os.path.join(model_dir, f"{model_name}.pkl"))

    with open(os.path.join(model_dir, f"{model_name}_metadata.json")) as f:
        metadata = json.load(f)

    return model, metadata, model_name


# ---------------------------------------------------------
# Rebuild the test split exactly as modelling did, re-apply
# the saved anomaly detectors, and return both the raw eval
# frame and the model-ready matrix (scaled if required,
# column-aligned to the model's expected features).
# ---------------------------------------------------------

def rebuild_eval_data(feature_file, metadata, model_dir,
                      target_col, timestamp_col, random_seed):

    df = load_feature_matrix(feature_file, target_col)
    df = sanitize_feature_names(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df, target_col, timestamp_col=timestamp_col, random_state=random_seed
    )

    # Re-apply anomaly features. We refit on train (identical seed
    # and data → identical detectors) so the eval columns match.
    X_train, X_val, X_test, _ = add_anomaly_features(
        X_train, X_val, X_test, random_state=random_seed
    )

    feature_cols = metadata["feature_columns"]

    # Align to the model's expected feature columns.
    X_eval = X_test.reindex(columns=feature_cols, fill_value=0)

    if metadata["scale_required"]:
        # Load the saved scaler and transform.
        scaler = joblib.load(
            os.path.join(model_dir, f"{metadata['model_name']}_scaler.pkl")
        )
        X_eval_model = pd.DataFrame(
            scaler.transform(X_eval),
            columns=feature_cols,
            index=X_eval.index
        )
    else:
        X_eval_model = X_eval

    return X_eval, y_test, X_eval_model


# ---------------------------------------------------------
# Native feature importance (tree models expose this).
# Returns None for models without it (e.g. LogReg, MLP).
# ---------------------------------------------------------

def native_feature_importance(model, feature_cols, output_dir):

    if not hasattr(model, "feature_importances_"):
        return None

    imp = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    df = imp.to_frame(name="native_importance")
    _save_table(df, output_dir, "feature_importance_native.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    imp.head(20).iloc[::-1].plot.barh(ax=ax)
    ax.set_title("Native feature importance (top 20)")
    ax.set_xlabel("importance")
    _save_fig(fig, output_dir, "feature_importance_native.png")

    return imp


# ---------------------------------------------------------
# Permutation importance — model-agnostic. Measures the
# drop in PR-AUC when each feature's values are shuffled.
# ---------------------------------------------------------

def permutation_feature_importance(model, X_eval, y_eval,
                                   feature_cols, output_dir, random_seed):

    result = permutation_importance(
        model, X_eval, y_eval,
        scoring="average_precision",
        n_repeats=10, random_state=random_seed, n_jobs=-1
    )

    imp = pd.Series(
        result.importances_mean, index=feature_cols
    ).sort_values(ascending=False)

    df = pd.DataFrame({
        "perm_importance_mean": result.importances_mean,
        "perm_importance_std":  result.importances_std
    }, index=feature_cols).sort_values("perm_importance_mean", ascending=False)

    _save_table(df, output_dir, "feature_importance_permutation.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    imp.head(20).iloc[::-1].plot.barh(ax=ax)
    ax.set_title("Permutation importance (top 20, by PR-AUC drop)")
    ax.set_xlabel("mean importance")
    _save_fig(fig, output_dir, "feature_importance_permutation.png")

    return imp


# ---------------------------------------------------------
# SHAP analysis. Uses TreeExplainer for tree models (fast,
# exact) and falls back gracefully for others.
# ---------------------------------------------------------

def shap_analysis(model, X_eval, metadata, output_dir, max_samples=500):

    try:
        import shap
    except ImportError:
        print("  shap not installed — run `pip install shap`. Skipping.")
        return False

    # Sample for speed on larger eval sets.
    X_sample = X_eval.sample(
        n=min(max_samples, len(X_eval)), random_state=42
    ) if len(X_eval) > max_samples else X_eval

    model_type = type(model).__name__

    try:
        if model_type in ("LGBMClassifier", "XGBClassifier",
                           "RandomForestClassifier", "CatBoostClassifier",
                           "HistGradientBoostingClassifier",
                           "ExtraTreesClassifier"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            # Binary classifiers sometimes return a list [neg, pos].
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            # Linear / MLP — use the model-agnostic explainer on a
            # small background sample.
            background = shap.sample(X_sample, min(100, len(X_sample)), random_state=42)
            explainer = shap.Explainer(model.predict_proba, background)
            shap_values = explainer(X_sample)[..., 1].values

        fig = plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
        fig.savefig(os.path.join(output_dir, "shap_summary.png"),
                    bbox_inches="tight", dpi=120)
        plt.close(fig)
        return True

    except Exception as e:
        print(f"  SHAP failed for {model_type}: {e}")
        return False


# ---------------------------------------------------------
# Partial dependence plots for the top features.
# ---------------------------------------------------------

def partial_dependence_plots(model, X_eval, features, output_dir):

    plot_dir = os.path.join(output_dir, "partial_dependence")
    saved = []

    for feat in features:
        if feat not in X_eval.columns:
            continue
        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            PartialDependenceDisplay.from_estimator(
                model, X_eval, [feat], ax=ax
            )
            ax.set_title(f"Partial dependence — {feat}")
            saved.append(_save_fig(fig, plot_dir, f"{feat}.png"))
        except Exception as e:
            print(f"  PDP failed for {feat}: {e}")

    return saved