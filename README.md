# Nova Pay Fraud Detection Prototype

This repository contains a prototype machine learning system designed to detect fraudulent financial transactions for Nova Pay.

The project demonstrates a full data science workflow including data exploration, feature engineering, model development, evaluation, and deployment through a lightweight application.

## Project Objectives

Build a fraud detection model capable of identifying suspicious financial transactions while minimizing false positives.

The project focuses on:

- Handling highly imbalanced fraud datasets
- Feature engineering for transaction behaviour
- Comparing multiple machine learning models
- Building an explainable fraud scoring system
- Creating a simple prototype interface for predictions

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
│
├── data
│ ├── raw
│ └── processed
│
├── notebooks
│
├── src
│ ├── data
│ ├── features
│ ├── models
│ └── utils
│
├── app
│
├── configs
│
├── models
│
├── requirements.txt
├── environment.yml
└── README.md



## Setup Instructions

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


## Reproducibility

All experiments use a fixed random seed defined in the configuration files.

## Author
- Dauda Agbonoga - Data Science Assessment Submission
