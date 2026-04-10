import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from api.schemas import CustomerInput, PredictionOutput
from src.preprocess import clean_data, encode_features, scale_features
from dotenv import load_dotenv

load_dotenv()

model = None
scaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root_dir, "models", "LogisticRegression.pkl")
    scaler_path = os.path.join(root_dir, "models", "scaler.pkl")

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise RuntimeError(f"Scaler not found at {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"Model loaded ✅")
    print(f"Scaler loaded ✅")
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability for telecom customers",
    version="1.0.0",
    lifespan=lifespan
)


def get_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    else:
        return "High"


def get_message(probability: float) -> str:
    if probability < 0.3:
        return "Customer is unlikely to churn. No immediate action needed."
    elif probability < 0.6:
        return "Customer shows some churn signals. Consider a retention offer."
    else:
        return "High churn risk! Immediate retention action recommended."


@app.get("/")
def root():
    return {"message": "Churn Prediction API is running ✅", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    try:
        data = pd.DataFrame([customer.model_dump()])

        # Add placeholders for pipeline compatibility
        data['TotalCharges'] = data['MonthlyCharges'] * data['tenure']
        data['customerID'] = 'API_INPUT'
        data['Churn'] = 'No'

        # Preprocess
        data = clean_data(data)
        data = encode_features(data)
        data, _ = scale_features(data, scaler=scaler, fit=False)
        data = data.drop(columns=['Churn'], errors='ignore')

        # Align columns with training data
        api_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(api_dir)
        columns_path = os.path.join(root_dir, "models", "feature_columns.pkl")
        expected_columns = joblib.load(columns_path)
        data = data.reindex(columns=expected_columns, fill_value=0)

        # Predict
        probability = float(model.predict_proba(data)[0][1])
        prediction = probability >= 0.5
        risk = get_risk_level(probability)
        message = get_message(probability)

        return PredictionOutput(
            churn_probability=round(probability, 4),
            churn_prediction=prediction,
            risk_level=risk,
            message=message
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))