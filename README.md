# Nova Pay Fraud Detection Prototype

This repository contains an end-to-end MLOps pipeline for detecting fraudulent transactions on the
NovaPay platform.

The project demonstrates a full data science workflow including data exploration, feature engineering, model development, evaluation, and deployment through a lightweight application.

## Project Objectives

Build a fraud detection model capable of identifying suspicious financial transactions while minimizing false positives.

The project focuses on:

- Handling highly imbalanced fraud datasets
- Feature engineering for transaction behaviour
- Comparing multiple machine learning models
- Building an explainable fraud scoring system
- Creating a simple prototype interface for predictions

## Pipeline overview

The pipeline is run as a sequence of stages, each executed via `run.py`:

| Stage | Command | Purpose |
|-------|---------|---------|
| Ingestion | `python run.py --step ingestion` | Schema validation, timestamp parsing, deduplication |
| Cleaning | `python run.py --step cleaning` | ML-based imputation, typo correction, missingness indicators |
| EDA | `python run.py --step eda` | Statistical summaries, fraud-rate analyses, dashboard JSON |
| Feature engineering | `python run.py --step feature_engineering` | Time features, customer aggregates, cohort z-scores, one-hot encoding |
| Modelling | `python run.py --step modelling` | Train + tune linear/tree/neural finalists with Optuna, CV-based selection |
| Explainability | `python run.py --step explainability` | Feature importance, SHAP, partial dependence |
| Prediction | `python run.py --step prediction` | Batch prediction with the winning model |

Or run the full pipeline:

```bash
python run.py --step all
```

## Tech Stack

Python libraries used include:

- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- catboost
- imbalanced-learn
- joblib
- streamlit
- matplotlib
- seaborn

## Project Structure

nova-pay-fraud-prototype
- │
- ├── data
- │ ├── raw
- │ └── processed
- │
- ├── notebooks
- │
- ├── src
- │ ├── data
- │ ├── features
- │ ├── models
- │ └── utils
- │
- ├── app
- │
- ├── configs
- │
- ├── models
- │
- ├── requirements.txt
- ├── environment.yml
- └── README.md


## Setup

```bash
python -m venv nova-pay-env
nova-pay-env\Scripts\activate            # Windows
source nova-pay-env/bin/activate         # macOS / Linux

pip install -r requirements.txt
```

### Instructions

Clone the repository:
- `git clone https://github.com/Dee-ui/nova-pay-fraud-prototype.git`
- `cd nova-pay-fraud-prototype`


Create environment:
- `conda env create -f environment.yml`
- `conda activate nova-pay-env`


Install dependencies:
- `pip install -r requirements.txt`


## Running the Project

Launch the Streamlit application:
- `streamlit run app/app.py`

## Experiment tracking

Every modelling run is logged to a local MLflow tracking store. View the
UI with:

```bash
mlflow ui
```

Then open `http://127.0.0.1:5000`.

## Data versioning

Data and model files are versioned with DVC. See `docs/dvc.md` for setup.

## Current model status

Best CV PR-AUC ~0.085 (baseline 0.015 → ~5x lift). Limited by data size:
145 fraud positives in training set. See `reports/explainability/` for
feature importance analysis.


## Reproducibility

All experiments use a fixed random seed defined in the configuration files.

## Author
- Dauda Agbonoga - Data Science Assessment Submission
