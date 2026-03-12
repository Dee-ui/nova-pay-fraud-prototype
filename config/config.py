"""
Central configuration file for the NovaPay Fraud Detection prototype.
All global project settings should be defined here.
"""

# -------------------------------------------------
# Reproducibility
# -------------------------------------------------

RANDOM_SEED = 42


# Data paths
# -------------------------------------------------

RAW_DATA_PATH = r"C:\Users\dauda.agbonoga\Documents\error\nova-pay-fraud-prototype\data\raw"
PROCESSED_DATA_PATH = r"C:\Users\dauda.agbonoga\Documents\error\nova-pay-fraud-prototype\data\processed"

# Raw inputs
TRANSACTION_FILE = r"C:\Users\dauda.agbonoga\Documents\error\nova-pay-fraud-prototype\data\raw\nova_pay_transcations (1).csv"
DATA_DICTIONARY_FILE = r"C:\Users\dauda.agbonoga\Documents\error\nova-pay-fraud-prototype\data\raw\data_dictionary (1).csv"

# Pipeline stage files
INGESTION_OUTPUT_FILE = r"C:\Users\dauda.agbonoga\Documents\error\nova-pay-fraud-prototype\data\processed\transactions_cleaned.csv"
PREPARED_DATA_FILE = r"C:\Users\dauda.agbonoga\Documents\error\nova-pay-fraud-prototype\data\processed\transactions_prepared.csv"


FEATURE_DATA_FILE = r"C:\Users\dauda.agbonoga\Documents\error\nova-pay-fraud-prototype\data\processed\transactions_feature_engineered.csv"

# -------------------------------------------------
# Model artifacts
# -------------------------------------------------

BEST_MODEL_FILE = r"C:\Users\dauda.agbonoga\Documents\error\nova-pay-fraud-prototype\models\best_fraud_model.pkl"

MODEL_METRICS_FILE = r"C:\Users\dauda.agbonoga\Documents\error\nova-pay-fraud-prototype\reports\model_metrics.csv"


# -------------------------------------------------
# Reports
# -------------------------------------------------

REPORTS_PATH = r"C:\Users\dauda.agbonoga\Documents\error\nova-pay-fraud-prototype\reports"


# -------------------------------------------------
# EDA outputs
# -------------------------------------------------

EDA_OUTPUT_PATH = r"C:\Users\dauda.agbonoga\Documents\error\nova-pay-fraud-prototype\data\processed\eda"

MODEL_OUTPUT_PATH = r"models/"


# -------------------------------------------------
# Model parameters
# -------------------------------------------------

TEST_SIZE = 0.2
CROSS_VALIDATION_FOLDS = 5


# -------------------------------------------------
# Fraud threshold
# -------------------------------------------------

FRAUD_PROBABILITY_THRESHOLD = 0.5