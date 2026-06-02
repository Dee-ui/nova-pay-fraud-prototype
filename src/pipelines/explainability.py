import os
import json
import joblib
import pandas as pd

from src.utils.explainability_utils import (
    load_winning_model,
    rebuild_eval_data,
    native_feature_importance,
    permutation_feature_importance,
    shap_analysis,
    partial_dependence_plots
)


TARGET_COLUMN = "is_fraud"
TIMESTAMP_COL = "timestamp"


def run_explainability(
    feature_file,
    model_dir,
    metrics_file,
    output_dir,
    random_seed=42
):

    print("=" * 60)
    print("STARTING EXPLAINABILITY STAGE")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Identify the winning model from the metrics CSV
    # ---------------------------------------------------------

    print("\n[1/6] Loading winning model and metadata...")
    model, metadata, model_name = load_winning_model(model_dir, metrics_file)
    print(f"  Winning model: {model_name}")
    print(f"  Scale required: {metadata['scale_required']}")
    print(f"  Feature count:  {len(metadata['feature_columns'])}")

    # ---------------------------------------------------------
    # 2. Rebuild the evaluation data the same way modelling did
    # ---------------------------------------------------------

    print("\n[2/6] Rebuilding evaluation data (test split + anomaly features)...")
    X_eval, y_eval, X_eval_model = rebuild_eval_data(
        feature_file, metadata, model_dir,
        TARGET_COLUMN, TIMESTAMP_COL, random_seed
    )
    print(f"  Evaluation rows: {X_eval.shape[0]}  features: {X_eval.shape[1]}")

    # ---------------------------------------------------------
    # 3. Native feature importance (tree models only)
    # ---------------------------------------------------------

    print("\n[3/6] Computing native feature importance...")
    native_imp = native_feature_importance(
        model, metadata["feature_columns"], output_dir
    )
    if native_imp is not None:
        print(f"  Saved native importance ({len(native_imp)} features)")
    else:
        print("  Model has no native importance — skipping (will use permutation).")

    # ---------------------------------------------------------
    # 4. Permutation importance (model-agnostic)
    # ---------------------------------------------------------

    print("\n[4/6] Computing permutation importance...")
    perm_imp = permutation_feature_importance(
        model, X_eval_model, y_eval,
        metadata["feature_columns"], output_dir, random_seed
    )
    print(f"  Saved permutation importance (top: {perm_imp.index[0]})")

    # ---------------------------------------------------------
    # 5. SHAP analysis
    # ---------------------------------------------------------

    print("\n[5/6] Computing SHAP values...")
    shap_ok = shap_analysis(
        model, X_eval_model, metadata, output_dir
    )
    print("  SHAP summary saved." if shap_ok else "  SHAP skipped for this model type.")

    # ---------------------------------------------------------
    # 6. Partial dependence for top features
    # ---------------------------------------------------------

    print("\n[6/6] Generating partial dependence plots...")
    top_features = perm_imp.head(5).index.tolist()
    pdp_paths = partial_dependence_plots(
        model, X_eval_model, top_features, output_dir
    )
    print(f"  Saved {len(pdp_paths)} partial-dependence plots.")

    print("\n" + "=" * 60)
    print("EXPLAINABILITY COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return {
        "status":        "success",
        "model_name":    model_name,
        "output_dir":    output_dir,
        "top_features":  top_features
    }