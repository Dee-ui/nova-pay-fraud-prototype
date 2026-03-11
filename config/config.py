"""
Central configuration file for the NovaPay Fraud Detection prototype.
All global project settings should be defined here.
"""

# -------------------------------------------------
# Reproducibility
# -------------------------------------------------

RANDOM_SEED = 42


# -------------------------------------------------
# Data paths
# -------------------------------------------------

RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"
MODEL_OUTPUT_PATH = "models/"


# -------------------------------------------------
# Model parameters
# -------------------------------------------------

TEST_SIZE = 0.2
CROSS_VALIDATION_FOLDS = 5


# -------------------------------------------------
# Fraud threshold
# -------------------------------------------------

FRAUD_PROBABILITY_THRESHOLD = 0.5