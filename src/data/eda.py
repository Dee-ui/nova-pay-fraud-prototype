# ===============================
# Exploratory Data Analysis
# ===============================

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(".."))

from config.config import PREPARED_DATA_FILE, EDA_OUTPUT_PATH, REPORTS_PATH

print("Loading prepared dataset for EDA...")

df = pd.read_csv(PREPARED_DATA_FILE)

print("Dataset loaded")
print(df.shape)

# ------------------------------------------------
# Normalize categorical values
# ------------------------------------------------

df["channel"] = df["channel"].str.lower().str.strip()
df["kyc_tier"] = df["kyc_tier"].str.lower().str.strip()
df["home_country"] = df["home_country"].str.upper().str.strip()

# Fix typos in KYC tier
kyc_corrections = {
    "standrd": "standard",
    "enhancd": "enhanced"
}

df["kyc_tier"] = df["kyc_tier"].replace(kyc_corrections)

# Fix typos in channel
channel_corrections = {
    "weeb": "web"
}

df["channel"] = df["channel"].replace(kyc_corrections)

TARGET_COL = "is_fraud"

numeric_cols = [
    "amount_usd",
    "fee",
    "device_trust_score",
    "ip_risk_score"
]

categorical_cols = [
    "channel",
    "kyc_tier",
    "home_country"
]

customer_behavior_cols = [
    "chargeback_history_count",
    "account_age_days"
]

# --------------------------------
# 1 Class balance
# --------------------------------

fraud_counts = df[TARGET_COL].value_counts()

print("Fraud vs Non Fraud Counts:")
print(fraud_counts)

plt.figure(figsize=(6,6))

plt.pie(
    fraud_counts,
    labels=["Non-Fraud","Fraud"],
    autopct="%1.2f%%",
    startangle=90
)

plt.title("Fraud vs Non-Fraud Distribution")
plt.show()

fraud_rate = fraud_counts[1] / fraud_counts.sum()

print(f"\nFraud rate: {fraud_rate:.4f}")

# --------------------------------
# 2 Numerical distributions
# --------------------------------

for col in numeric_cols:

    fig, axes = plt.subplots(1,2, figsize=(12,4))

    sns.histplot(df[col], kde=True, ax=axes[0])
    axes[0].set_title(f"{col} Distribution")

    sns.boxplot(
        x=TARGET_COL,
        y=col,
        data=df,
        ax=axes[1]
    )

    axes[1].set_title(f"{col} by Fraud Label")

    plt.tight_layout()
    plt.show()

# --------------------------------
# 3 Categorical fraud rate
# --------------------------------

for col in categorical_cols:

    pivot = (
        df.groupby(col)[TARGET_COL]
        .agg(["count","mean"])
        .rename(columns={"mean":"fraud_rate"})
        .sort_values("fraud_rate", ascending=False)
    )

    print(f"\n{col} fraud statistics")
    print(pivot)

    plt.figure(figsize=(8,4))

    sns.barplot(
        x=pivot.index,
        y=pivot["fraud_rate"]
    )

    plt.title(f"Fraud Rate by {col}")
    plt.ylabel("Fraud Rate")

    plt.xticks(rotation=45)

    plt.show()

# --------------------------------
# 4 Time patterns
# --------------------------------

hourly_fraud = df.groupby("txn_hour")[TARGET_COL].mean()
daily_fraud = df.groupby("txn_day_of_week")[TARGET_COL].mean()

plt.figure(figsize=(8,4))

sns.lineplot(x=hourly_fraud.index, y=hourly_fraud.values)

plt.title("Fraud Rate by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Fraud Rate")

plt.show()

plt.figure(figsize=(8,4))

sns.barplot(x=daily_fraud.index, y=daily_fraud.values)

plt.title("Fraud Rate by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Fraud Rate")

plt.show()

# --------------------------------
# 5 Customer behavior analysis
# --------------------------------

for col in customer_behavior_cols:

    plt.figure(figsize=(8,4))

    sns.boxplot(
        x=TARGET_COL,
        y=col,
        data=df
    )

    plt.title(f"{col} vs Fraud")

    plt.show()

# ------------------------------------------------
# Export results
# ------------------------------------------------

os.makedirs(EDA_OUTPUT_PATH, exist_ok=True)
os.makedirs(REPORTS_PATH, exist_ok=True)

fraud_channel = df.groupby("channel")["is_fraud"].agg(["count","mean"])

fraud_channel.to_csv(
    os.path.join(EDA_OUTPUT_PATH, "fraud_by_channel.csv")
)

summary = f"""
EDA SUMMARY
-----------

Total transactions: {len(df)}

Fraud count: {df['is_fraud'].sum()}
Fraud rate: {df['is_fraud'].mean():.4f}

Key observations:
- Review numeric distribution plots
- Review categorical fraud rate plots
- Review time based fraud plots
"""

with open(os.path.join(REPORTS_PATH, "eda_summary.txt"), "w") as f:
    f.write(summary)

print("EDA summary saved.")