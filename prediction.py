# ==========================================================
# prediction.py
# Customer Churn Prediction Module
# ==========================================================

# Import required libraries
import joblib
import pandas as pd
from pathlib import Path


# ==========================================================
# Load the Trained Pipeline
# ==========================================================

# Load the saved machine learning pipeline

MODEL_PATH = Path(__file__).parent / "churn_pipeline.pkl"

pipeline = joblib.load(MODEL_PATH)


# ==========================================================
# Prediction Function
# ==========================================================

def predict_churn(customer_data):
    """
    Predict whether a customer will churn.

    Parameters:
        customer_data (dict):
            Dictionary containing customer information.

    Returns:
        prediction (str):
            Churn or No Churn

        probability (float):
            Prediction confidence (%)
    """

    # Convert dictionary into DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Make prediction
    prediction = pipeline.predict(customer_df)[0]

    # Prediction probability
    probability = pipeline.predict_proba(customer_df)[0]

    # Convert prediction to readable text
    if prediction == 1:
        result = "Customer is likely to Churn"
        confidence = probability[1] * 100
    else:
        result = "Customer is likely to Stay"
        confidence = probability[0] * 100

    return result, round(confidence, 2)