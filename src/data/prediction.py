import joblib
import pandas as pd

def predict_transactions(data_file, model_file):

    df = pd.read_csv(data_file)

    model = joblib.load(model_file)

    X = df.drop(columns=["is_fraud"], errors="ignore")

    preds = model.predict(X)

    df["fraud_prediction"] = preds

    return df