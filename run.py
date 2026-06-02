"""
run.py — Single entry point for the NovaPay Fraud Detection pipeline.

Steps:
    python run.py --step ingestion
    python run.py --step cleaning
    python run.py --step eda
    python run.py --step feature_engineering
    python run.py --step modelling
    python run.py --step explainability
    python run.py --step prediction
    python run.py --step all
"""

import argparse


def run_ingestion_step():
    from src.pipelines.ingestion import run_ingestion
    from config.config import TRANSACTION_FILE, DATA_DICTIONARY_FILE, PROCESSED_DATA_PATH
    print("=" * 40); print("STEP: INGESTION"); print("=" * 40)
    results = run_ingestion(
        input_file=TRANSACTION_FILE,
        dictionary_file=DATA_DICTIONARY_FILE,
        output_path=PROCESSED_DATA_PATH
    )
    print("\nReturned Results:"); print(results)
    return results


def run_cleaning_step():
    from src.pipelines.cleaning import run_cleaning
    from config.config import INGESTION_OUTPUT_FILE, CLEANING_OUTPUT_FILE
    print("=" * 40); print("STEP: CLEANING"); print("=" * 40)
    results = run_cleaning(
        input_file=INGESTION_OUTPUT_FILE,
        output_file=CLEANING_OUTPUT_FILE
    )
    print("\nReturned Results:"); print(results)
    return results


def run_eda_step():
    from src.pipelines.eda import run_eda
    from config.config import CLEANING_OUTPUT_FILE, EDA_OUTPUT_PATH
    print("=" * 40); print("STEP: EDA"); print("=" * 40)
    results = run_eda(
        input_file=CLEANING_OUTPUT_FILE,
        output_dir=EDA_OUTPUT_PATH
    )
    print("\nReturned Results:")
    print({k: v for k, v in results.items() if k != "summary_payload"})
    return results


def run_feature_engineering_step():
    from src.pipelines.feature_engineering import run_feature_engineering
    from config.config import CLEANING_OUTPUT_FILE, FEATURE_DATA_FILE
    print("=" * 40); print("STEP: FEATURE ENGINEERING"); print("=" * 40)
    results = run_feature_engineering(
        input_file=CLEANING_OUTPUT_FILE,
        output_file=FEATURE_DATA_FILE
    )
    print("\nReturned Results:"); print(results)
    return results


def run_modelling_step():
    from src.pipelines.modelling import run_modelling
    from config.config import (
        FEATURE_DATA_FILE, MODEL_OUTPUT_PATH, MODEL_METRICS_FILE, RANDOM_SEED
    )
    print("=" * 40); print("STEP: MODELLING"); print("=" * 40)
    results = run_modelling(
        input_file=FEATURE_DATA_FILE,
        model_output_dir=MODEL_OUTPUT_PATH,
        metrics_output_file=MODEL_METRICS_FILE,
        random_seed=RANDOM_SEED
    )
    print("\nReturned Results:"); print(results)
    return results


def run_explainability_step():
    from src.pipelines.explainability import run_explainability
    from config.config import (
        FEATURE_DATA_FILE, MODEL_OUTPUT_PATH, MODEL_METRICS_FILE,
        EXPLAINABILITY_OUTPUT_PATH, RANDOM_SEED
    )
    print("=" * 40); print("STEP: EXPLAINABILITY"); print("=" * 40)
    results = run_explainability(
        feature_file=FEATURE_DATA_FILE,
        model_dir=MODEL_OUTPUT_PATH,
        metrics_file=MODEL_METRICS_FILE,
        output_dir=EXPLAINABILITY_OUTPUT_PATH,
        random_seed=RANDOM_SEED
    )
    print("\nReturned Results:"); print(results)
    return results


def run_prediction_step():
    from src.pipelines.prediction import predict_transactions
    from config.config import (
        FEATURE_DATA_FILE, MODEL_OUTPUT_PATH, MODEL_METRICS_FILE,
        PREDICTIONS_OUTPUT_FILE
    )
    import pandas as pd
    print("=" * 40); print("STEP: PREDICTION (batch demo)"); print("=" * 40)

    # Identify the winning model from the metrics CSV.
    metrics = pd.read_csv(MODEL_METRICS_FILE)
    winner = metrics[metrics["is_winner"]].iloc[0]["model"]

    results = predict_transactions(
        feature_file=FEATURE_DATA_FILE,
        model_dir=MODEL_OUTPUT_PATH,
        model_name=winner,
        output_file=PREDICTIONS_OUTPUT_FILE
    )
    print("\nSample predictions:")
    print(results.head())
    return results


def main():

    parser = argparse.ArgumentParser(
        description="NovaPay Fraud Detection Pipeline Runner"
    )
    parser.add_argument(
        "--step", type=str, required=True,
        choices=[
            "ingestion", "cleaning", "eda", "feature_engineering",
            "modelling", "explainability", "prediction", "all"
        ],
        help=("Pipeline step. Options: ingestion | cleaning | eda | "
              "feature_engineering | modelling | explainability | "
              "prediction | all")
    )
    args = parser.parse_args()

    if args.step == "ingestion":
        run_ingestion_step()
    elif args.step == "cleaning":
        run_cleaning_step()
    elif args.step == "eda":
        run_eda_step()
    elif args.step == "feature_engineering":
        run_feature_engineering_step()
    elif args.step == "modelling":
        run_modelling_step()
    elif args.step == "explainability":
        run_explainability_step()
    elif args.step == "prediction":
        run_prediction_step()
    elif args.step == "all":
        run_ingestion_step()
        run_cleaning_step()
        run_eda_step()
        run_feature_engineering_step()
        run_modelling_step()
        run_explainability_step()
        run_prediction_step()


if __name__ == "__main__":
    main()