# ============================================
# Modeling and Evaluation
# ============================================

import sys
import os
import pandas as pd
import numpy as np
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from sklearn.model_selection import RandomizedSearchCV


# ============================================
# Import config
# ============================================

sys.path.append(os.path.abspath(".."))

from config.config import (
    FEATURE_DATA_FILE,
    BEST_MODEL_FILE,
    MODEL_METRICS_FILE,
    RANDOM_SEED
)


# ============================================
# Load dataset
# ============================================

print("Loading feature engineered dataset...")

df = pd.read_csv(FEATURE_DATA_FILE, low_memory=False)

print("Dataset shape:", df.shape)

TARGET_COL = "is_fraud"


# ============================================
# Fix datatypes
# ============================================

print("\nFixing datatypes...")

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# force numeric conversion where possible
for col in df.columns:
    if col not in ["timestamp", TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="ignore")


# ============================================
# Detect categorical columns properly
# ============================================

print("\nDetecting categorical columns...")

categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

print("Categorical columns:", categorical_cols)


# ============================================
# One-hot encode categoricals
# ============================================

if len(categorical_cols) > 0:

    print("Applying one-hot encoding...")

    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=False
    )

print("Dataset shape after encoding:", df.shape)


# ============================================
# Clean feature names (LightGBM requirement)
# ============================================

print("\nCleaning feature names...")

df.columns = [
    re.sub('[^A-Za-z0-9_]+', '_', col)
    for col in df.columns
]


# ============================================
# Dataset split
# ============================================

print("\nCreating dataset split...")

if "timestamp" in df.columns:

    print("Using time based split")

    df = df.sort_values("timestamp")

    train_size = int(len(df) * 0.70)
    val_size = int(len(df) * 0.85)

    train = df.iloc[:train_size]
    val = df.iloc[train_size:val_size]
    test = df.iloc[val_size:]

    drop_cols = ["timestamp"]

else:

    print("Timestamp not found → using random split")

    train, temp = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df[TARGET_COL]
    )

    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp[TARGET_COL]
    )

    drop_cols = []

print("Train:", train.shape)
print("Validation:", val.shape)
print("Test:", test.shape)


# ============================================
# Feature / target split
# ============================================

X_train = train.drop(columns=[TARGET_COL] + drop_cols, errors="ignore")
y_train = train[TARGET_COL]

X_val = val.drop(columns=[TARGET_COL] + drop_cols, errors="ignore")
y_val = val[TARGET_COL]

X_test = test.drop(columns=[TARGET_COL] + drop_cols, errors="ignore")
y_test = test[TARGET_COL]


# ============================================
# Feature alignment
# ============================================

print("\nAligning feature columns...")

X_train, X_val = X_train.align(X_val, join="left", axis=1, fill_value=0)
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

print("Training features:", X_train.shape)


# ============================================
# Scale features for Logistic Regression
# ============================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# ============================================
# Train baseline models
# ============================================

print("\nTraining baseline models...")

sample_frac = 0.30

sample_idx = X_train.sample(frac=sample_frac, random_state=RANDOM_SEED).index

X_sample = X_train.loc[sample_idx]
y_sample = y_train.loc[sample_idx]

X_sample_scaled = scaler.transform(X_sample)

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=3000, class_weight="balanced"),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced",
            random_state=RANDOM_SEED
        ),

    "XGBoost":
        xgb.XGBClassifier(
            eval_metric="logloss",
            scale_pos_weight=10,
            random_state=RANDOM_SEED
        ),

    "LightGBM":
        lgb.LGBMClassifier(
            class_weight="balanced",
            random_state=RANDOM_SEED
        ),

    "CatBoost":
        CatBoostClassifier(
            verbose=0,
            random_state=RANDOM_SEED
        )
}

results = []

for name, model in models.items():

    print(f"\nTraining {name}")

    if name == "Logistic Regression":
        model.fit(X_sample_scaled, y_sample)
        preds = model.predict(X_val_scaled)
        probs = model.predict_proba(X_val_scaled)[:, 1]

    else:
        model.fit(X_sample, y_sample)
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]

    roc = roc_auc_score(y_val, probs)
    pr = average_precision_score(y_val, probs)
    f1 = f1_score(y_val, preds)

    results.append({
        "model": name,
        "roc_auc": roc,
        "pr_auc": pr,
        "f1": f1
    })


results_df = pd.DataFrame(results)

print("\nModel comparison")
print(results_df)


# ============================================
# Best model selection
# ============================================

best_model_name = results_df.sort_values(
    "pr_auc",
    ascending=False
).iloc[0]["model"]

print("\nBest baseline model:", best_model_name)

best_model = models[best_model_name]


# ============================================
# Hyperparameter tuning
# ============================================

print("\nRunning hyperparameter tuning...")

if best_model_name in ["XGBoost", "LightGBM"]:

    param_grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1]
    }

elif best_model_name == "Random Forest":

    param_grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [None, 10, 20]
    }

elif best_model_name == "Logistic Regression":

    param_grid = {
        "C": [0.01, 0.1, 1, 10]
    }

else:

    param_grid = {}

if len(param_grid) > 0:

    search = RandomizedSearchCV(
        best_model,
        param_distributions=param_grid,
        n_iter=10,
        scoring="average_precision",
        cv=3,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    if best_model_name == "Logistic Regression":
        search.fit(X_train_scaled, y_train)
    else:
        search.fit(X_train, y_train)

    best_model = search.best_estimator_

    print("Best parameters:", search.best_params_)


# ============================================
# Final evaluation
# ============================================

print("\nEvaluating final model...")

if best_model_name == "Logistic Regression":
    preds = best_model.predict(X_test_scaled)
    probs = best_model.predict_proba(X_test_scaled)[:, 1]
else:
    preds = best_model.predict(X_test)
    probs = best_model.predict_proba(X_test)[:, 1]

metrics = {

    "ROC_AUC": roc_auc_score(y_test, probs),
    "PR_AUC": average_precision_score(y_test, probs),
    "Precision": precision_score(y_test, preds),
    "Recall": recall_score(y_test, preds),
    "F1": f1_score(y_test, preds)
}

print("\nFinal Model Performance")

for k, v in metrics.items():
    print(f"{k}: {v:.4f}")


# ============================================
# Save trained model
# ============================================

os.makedirs(os.path.dirname(BEST_MODEL_FILE), exist_ok=True)

joblib.dump(best_model, BEST_MODEL_FILE)

print("\nModel saved to:", BEST_MODEL_FILE)


# ============================================
# Save metrics
# ============================================

metrics_df = pd.DataFrame([metrics])

os.makedirs(os.path.dirname(MODEL_METRICS_FILE), exist_ok=True)

metrics_df.to_csv(MODEL_METRICS_FILE, index=False)

print("Metrics exported to:", MODEL_METRICS_FILE)