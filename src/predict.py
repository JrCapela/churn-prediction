import os
import sys
import joblib
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import clean_data, encode_features, scale_features


def load_artifacts():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = joblib.load(os.path.join(root_dir, 'models', 'LogisticRegression.pkl'))
    scaler = joblib.load(os.path.join(root_dir, 'models', 'scaler.pkl'))
    feature_columns = joblib.load(os.path.join(root_dir, 'models', 'feature_columns.pkl'))
    return model, scaler, feature_columns


def predict_customer(customer: dict) -> dict:
    model, scaler, feature_columns = load_artifacts()

    data = pd.DataFrame([customer])
    data['TotalCharges'] = data['MonthlyCharges'] * data['tenure']
    data['customerID'] = 'PREDICT_INPUT'
    data['Churn'] = 'No'

    data = clean_data(data)
    data = encode_features(data)
    data, _ = scale_features(data, scaler=scaler, fit=False)
    data = data.drop(columns=['Churn'], errors='ignore')
    data = data.reindex(columns=feature_columns, fill_value=0)

    probability = float(model.predict_proba(data)[0][1])
    prediction = probability >= 0.5

    if probability < 0.3:
        risk = "Low"
        message = "Customer is unlikely to churn."
    elif probability < 0.6:
        risk = "Medium"
        message = "Customer shows some churn signals."
    else:
        risk = "High"
        message = "High churn risk! Immediate action recommended."

    return {
        "churn_probability": round(probability, 4),
        "churn_prediction": prediction,
        "risk_level": risk,
        "message": message
    }


if __name__ == "__main__":
    sample = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0
    }

    result = predict_customer(sample)
    print("\n=== Churn Prediction ===")
    for key, value in result.items():
        print(f"{key}: {value}")