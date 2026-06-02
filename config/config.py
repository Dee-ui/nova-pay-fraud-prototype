"""
Central configuration file for the NovaPay Fraud Detection prototype.
All global project settings should be defined here.
"""

import os

# -------------------------------------------------
# Project root (resolves all paths dynamically)
# -------------------------------------------------

PROJECT_ROOT = r"C:\Users\Dauda Agbonoga\OneDrive - Venture Garden Group\Documents\my\nova-pay-fraud-project"


# -------------------------------------------------
# Reproducibility
# -------------------------------------------------

RANDOM_SEED = 42


# -------------------------------------------------
# Data paths
# -------------------------------------------------

RAW_DATA_PATH       = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")


# -------------------------------------------------
# Raw inputs
# -------------------------------------------------

TRANSACTION_FILE      = os.path.join(RAW_DATA_PATH, "nova_pay_transcations (1).csv")
DATA_DICTIONARY_FILE  = os.path.join(RAW_DATA_PATH, "data_dictionary (1).csv")


# -------------------------------------------------
# Pipeline stage output files
# -------------------------------------------------

# Output of the ingestion layer
INGESTION_OUTPUT_FILE = os.path.join(PROCESSED_DATA_PATH, "transactions_ingested.csv")

# Output of the cleaning layer
CLEANING_OUTPUT_FILE  = os.path.join(PROCESSED_DATA_PATH, "transactions_cleaned.csv")

# Output of the feature engineering layer
FEATURE_DATA_FILE     = os.path.join(PROCESSED_DATA_PATH, "transactions_feature_engineered.csv")


# -------------------------------------------------
# Model artifacts
# -------------------------------------------------

MODEL_OUTPUT_PATH  = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_FILE    = os.path.join(MODEL_OUTPUT_PATH, "best_fraud_model.pkl")


# -------------------------------------------------
# Reports
# -------------------------------------------------

REPORTS_PATH       = os.path.join(PROJECT_ROOT, "reports")
MODEL_METRICS_FILE = os.path.join(REPORTS_PATH, "model_metrics.csv")


# -------------------------------------------------
# EDA outputs
# -------------------------------------------------

EDA_OUTPUT_PATH    = os.path.join(PROCESSED_DATA_PATH, "eda")

# -------------------------------------------------
# Explainability outputs
# -------------------------------------------------

EXPLAINABILITY_OUTPUT_PATH = os.path.join(REPORTS_PATH, "explainability")


# -------------------------------------------------
# Prediction outputs
# -------------------------------------------------

PREDICTIONS_OUTPUT_FILE = os.path.join(PROCESSED_DATA_PATH, "predictions.csv")

# -------------------------------------------------
# Model parameters
# -------------------------------------------------

TEST_SIZE              = 0.2
CROSS_VALIDATION_FOLDS = 5


# -------------------------------------------------
# Fraud detection threshold
# -------------------------------------------------

FRAUD_PROBABILITY_THRESHOLD = 0.5