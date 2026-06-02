import os
import pathlib
import pandas as pd
import mlflow

from datetime import datetime

from src.utils.modelling_utils import (
    load_feature_matrix,
    sanitize_feature_names,
    split_train_val_test,
    add_anomaly_features,
    fit_preprocessing,
    apply_preprocessing,
    build_linear_candidates,
    build_tree_candidates,
    build_neural_candidates,
    train_and_score_candidates,
    tune_finalist_optuna,
    find_best_threshold,
    compute_metrics,
    save_model_bundle,
    plot_pr_curve
)


TARGET_COLUMN = "is_fraud"
TIMESTAMP_COL = "timestamp"

CANDIDATE_SAMPLE_FRAC = 0.30
OPTUNA_N_TRIALS = 30

MLFLOW_EXPERIMENT_NAME = "nova-pay-fraud"


def _scale_pos_weight_for(y_train):
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    return float(neg) / max(float(pos), 1.0)


def _setup_mlflow(tracking_uri=None):
    if tracking_uri is None:
        project_root = pathlib.Path(__file__).parent.parent.parent
        db_path = project_root / "mlflow.db"
        tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def run_modelling(
    input_file,
    model_output_dir,
    metrics_output_file,
    random_seed=42,
    mlflow_tracking_uri=None
):

    _setup_mlflow(mlflow_tracking_uri)

    print("=" * 60)
    print("STARTING MODELING STAGE")
    print("=" * 60)

    # Capture a single timestamp for this entire pipeline run.
    # Used in both the parent and child run names so they all
    # share the same timestamp — making it easy to group them.
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    with mlflow.start_run(
        run_name=f"modelling_pipeline_{run_timestamp}"
    ) as parent_run:

        mlflow.set_tag("pipeline_stage", "modelling")
        mlflow.set_tag("dataset_file", os.path.basename(input_file))
        mlflow.log_param("random_seed", random_seed)
        mlflow.log_param("candidate_sample_frac", CANDIDATE_SAMPLE_FRAC)
        mlflow.log_param("optuna_n_trials", OPTUNA_N_TRIALS)

        # -----------------------------------------------------
        # 1. Load and validate
        # -----------------------------------------------------

        print("\n[1/9] Loading feature matrix...")
        df = load_feature_matrix(input_file, TARGET_COLUMN)
        print(f"  Loaded {df.shape[0]} rows × {df.shape[1]} columns")
        df = sanitize_feature_names(df)

        mlflow.log_param("n_rows", df.shape[0])
        mlflow.log_param("n_columns_pre_anomaly", df.shape[1])

        # -----------------------------------------------------
        # 2. Split
        # -----------------------------------------------------

        print("\n[2/9] Splitting train / val / test...")
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
            df, TARGET_COLUMN, timestamp_col=TIMESTAMP_COL, random_state=random_seed
        )
        print(f"  Train: {X_train.shape}  positives={int(y_train.sum())}")
        print(f"  Val:   {X_val.shape}  positives={int(y_val.sum())}")
        print(f"  Test:  {X_test.shape}  positives={int(y_test.sum())}")

        mlflow.log_metric("train_positives", int(y_train.sum()))
        mlflow.log_metric("val_positives",   int(y_val.sum()))
        mlflow.log_metric("test_positives",  int(y_test.sum()))
        mlflow.log_metric("train_fraud_rate", float(y_train.mean()))

        # -----------------------------------------------------
        # 3. Anomaly features
        # -----------------------------------------------------

        print("\n[3/9] Adding unsupervised anomaly-score features...")
        X_train, X_val, X_test, anomaly_detectors = add_anomaly_features(
            X_train, X_val, X_test, random_state=random_seed
        )

        # -----------------------------------------------------
        # 4. Preprocessing
        # -----------------------------------------------------

        print("\n[4/9] Fitting preprocessing...")
        preproc = fit_preprocessing(X_train)
        print(f"  Dropped {len(preproc['dropped_cols'])} constant columns")

        X_val_raw_arr     = apply_preprocessing(X_val,  preproc, scale=False)
        X_val_scaled_arr  = apply_preprocessing(X_val,  preproc, scale=True)
        X_test_raw_arr    = apply_preprocessing(X_test, preproc, scale=False)
        X_test_scaled_arr = apply_preprocessing(X_test, preproc, scale=True)

        X_train_raw_df = X_train[preproc["kept_cols"]]
        X_val_raw_df   = X_val[preproc["kept_cols"]]
        X_test_raw_df  = X_test[preproc["kept_cols"]]

        scale_pos_weight = _scale_pos_weight_for(y_train)

        # -----------------------------------------------------
        # 5. Candidate selection per family
        # -----------------------------------------------------

        print("\n[5/9] Candidate selection...")
        all_family_results = {}

        for family_label, build_fn, args in [
            ("linear", build_linear_candidates, (random_seed,)),
            ("tree",   build_tree_candidates,   (random_seed, scale_pos_weight)),
            ("neural", build_neural_candidates, (random_seed,))
        ]:
            print(f"\n  {family_label.upper()} FAMILY")
            cands = build_fn(*args)
            results = train_and_score_candidates(
                cands, X_train_raw_df, y_train,
                X_val_raw_df, X_val_scaled_arr, y_val,
                sample_frac=CANDIDATE_SAMPLE_FRAC, random_state=random_seed
            )
            print(f"\n  {family_label.capitalize()} leaderboard:")
            print(results[["candidate", "pr_auc", "roc_auc", "f1"]].to_string(index=False))
            all_family_results[family_label] = (cands, results)

        linear_cands, linear_results = all_family_results["linear"]
        tree_cands,   tree_results   = all_family_results["tree"]
        neural_cands, neural_results = all_family_results["neural"]

        linear_winner_name = linear_results.iloc[0]["candidate"]
        tree_winner_name   = tree_results.iloc[0]["candidate"]
        neural_winner_name = neural_results.iloc[0]["candidate"]

        print(f"\n  Family winners: linear={linear_winner_name}, "
              f"tree={tree_winner_name}, neural={neural_winner_name}")

        # -----------------------------------------------------
        # 6/7/8. Tune, threshold, test — each in a CHILD run
        # -----------------------------------------------------

        use_time_cv = TIMESTAMP_COL in df.columns
        finalists = {}

        for family_name, winner_name, candidates_dict in [
            ("linear", linear_winner_name, linear_cands),
            ("tree",   tree_winner_name,   tree_cands),
            ("neural", neural_winner_name, neural_cands)
        ]:
            # Child run name includes family, model name, and the
            # same timestamp as the parent so they group visually
            # in the UI. CV score is added as a tag after tuning.
            with mlflow.start_run(
                run_name=f"{family_name}_{winner_name}_{run_timestamp}",
                nested=True
            ):

                mlflow.set_tag("family", family_name)
                mlflow.set_tag("model_name", winner_name)

                print(f"\n  Tuning {family_name} finalist: {winner_name}")
                _, opts = candidates_dict[winner_name]
                scale = opts["scale"]

                tuned_model, best_params, cv_score = tune_finalist_optuna(
                    name=winner_name, scale=scale,
                    X_train=X_train_raw_df, y_train=y_train,
                    scaler_full=preproc["scaler"],
                    scale_pos_weight=scale_pos_weight,
                    use_time_cv=use_time_cv, n_trials=OPTUNA_N_TRIALS,
                    random_state=random_seed
                )

                # Now that cv_score is known, tag it for easy
                # identification directly in the UI runs table.
                mlflow.set_tag("cv_pr_auc_summary", f"{cv_score:.4f}")

                mlflow.log_params(best_params)
                mlflow.log_metric("cv_pr_auc", cv_score)

                # Threshold tuning on val
                X_val_input = X_val_scaled_arr if scale else X_val_raw_df.values
                val_proba = tuned_model.predict_proba(X_val_input)[:, 1]
                threshold_info = find_best_threshold(y_val, val_proba, target="f1")
                threshold = threshold_info["threshold"]
                mlflow.log_metric("threshold", threshold)
                mlflow.log_metric("val_f1_at_threshold", threshold_info["f1"])

                # Test eval (reported, not used for selection)
                X_test_input = X_test_scaled_arr if scale else X_test_raw_df.values
                test_proba = tuned_model.predict_proba(X_test_input)[:, 1]
                test_pred  = (test_proba >= threshold).astype(int)
                test_metrics = compute_metrics(y_test, test_pred, test_proba)

                for k, v in test_metrics.items():
                    mlflow.log_metric(f"test_{k}", v)

                # PR curve as an artifact
                pr_path = plot_pr_curve(y_test, test_proba, model_output_dir,
                                        winner_name)
                mlflow.log_artifact(pr_path)

                finalists[family_name] = {
                    "name": winner_name, "model": tuned_model, "scale": scale,
                    "best_params": best_params, "cv_pr_auc": cv_score,
                    "threshold": threshold,
                    "val_thresh_f1": threshold_info["f1"],
                    "test_metrics": test_metrics
                }

                print(f"  CV PR-AUC={cv_score:.4f}, "
                      f"test PR-AUC={test_metrics['pr_auc']:.4f}")

        # -----------------------------------------------------
        # 9. Select winner (CV PR-AUC only) and save artifacts
        # -----------------------------------------------------

        print("\n[9/9] Selecting winner (by CV PR-AUC)...")

        ranked = sorted(
            finalists.items(),
            key=lambda kv: kv[1]["cv_pr_auc"],
            reverse=True
        )
        overall_winner_family = ranked[0][0]
        overall_winner_info   = ranked[0][1]

        print(f"  Winner: {overall_winner_info['name']} "
              f"({overall_winner_family}) — "
              f"CV PR-AUC={overall_winner_info['cv_pr_auc']:.4f}")

        # Tag the parent run with the winner.
        mlflow.set_tag("overall_winner", overall_winner_info["name"])
        mlflow.set_tag("overall_winner_family", overall_winner_family)
        mlflow.set_tag("winner_cv_pr_auc_summary",
                       f"{overall_winner_info['cv_pr_auc']:.4f}")
        mlflow.log_metric("winner_cv_pr_auc", overall_winner_info["cv_pr_auc"])
        mlflow.log_metric("winner_test_pr_auc",
                          overall_winner_info["test_metrics"]["pr_auc"])

        # Save every finalist to disk (existing behaviour).
        saved_paths = {}
        for family_name, info in finalists.items():
            paths = save_model_bundle(
                output_dir=model_output_dir,
                name=info["name"],
                model=info["model"],
                scaler=preproc["scaler"],
                feature_cols=preproc["kept_cols"],
                threshold=info["threshold"],
                metrics=info["test_metrics"],
                params=info["best_params"],
                scale_required=info["scale"],
                anomaly_detectors=anomaly_detectors
            )
            saved_paths[family_name] = paths

        # Log the winning model to the MLflow model registry.
        try:
            mlflow.sklearn.log_model(
                sk_model=overall_winner_info["model"],
                name="winner_model",
                registered_model_name="nova_pay_fraud_winner"
            )
        except Exception as e:
            print(f"  (MLflow model registry skipped: {e})")

        # Combined metrics CSV
        metrics_rows = []
        for family_name, info in finalists.items():
            row = {
                "family":     family_name,
                "model":      info["name"],
                "threshold":  info["threshold"],
                "cv_pr_auc":  info["cv_pr_auc"],
                "is_winner":  family_name == overall_winner_family,
                **info["test_metrics"]
            }
            metrics_rows.append(row)

        metrics_df = pd.DataFrame(metrics_rows)
        os.makedirs(os.path.dirname(metrics_output_file), exist_ok=True)
        metrics_df.to_csv(metrics_output_file, index=False)
        mlflow.log_artifact(metrics_output_file)

        for family_name, (_, results_df) in all_family_results.items():
            path = os.path.join(model_output_dir, f"candidates_{family_name}.csv")
            results_df.to_csv(path, index=False)
            mlflow.log_artifact(path)

        print("\n" + "=" * 60)
        print("MODELING COMPLETED SUCCESSFULLY")
        print(f"MLflow run ID: {parent_run.info.run_id}")
        print("=" * 60)

    return {
        "status":                "success",
        "overall_winner":        overall_winner_info["name"],
        "overall_winner_family": overall_winner_family,
        "finalists":             {k: {
            "name":         v["name"],
            "threshold":    v["threshold"],
            "cv_pr_auc":    v["cv_pr_auc"],
            "test_metrics": v["test_metrics"]
        } for k, v in finalists.items()},
        "model_dir":             model_output_dir,
        "metrics_file":          metrics_output_file,
        "saved_paths":           saved_paths,
        "mlflow_run_id":         parent_run.info.run_id
    }