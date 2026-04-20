# Churn Prediction API 🎯

A production-grade machine learning project that predicts customer churn for telecom companies. Built with a complete MLOps pipeline including data versioning, experiment tracking, REST API, automated tests, containerization, and CI/CD.

## 📊 Problem

Telecom companies lose billions annually from customer churn. This project predicts **which customers are likely to cancel** before they do, enabling proactive retention actions.

- **Dataset:** Telco Customer Churn (7,043 customers, 21 features)
- **Model:** Logistic Regression (AUC-ROC: 0.839)
- **Churn rate:** 26.5% (imbalanced dataset)

## 🏗️ Architecture
churn-prediction/
├── api/                    # FastAPI REST API
├── src/                    # Core ML code
│   ├── preprocess.py       # Data preprocessing pipeline
│   ├── train.py            # Model training + MLflow tracking
│   ├── evaluate.py         # Model evaluation + SHAP
│   └── predict.py          # Standalone prediction script
├── notebooks/              # EDA and experiments
├── tests/                  # Pytest unit and integration tests
├── models/                 # Trained models (versioned with DVC)
├── data/                   # Dataset (versioned with DVC)
├── .github/workflows/      # CI/CD pipelines
├── Dockerfile              # API container
└── docker-compose.yml      # API + MLflow orchestration

## 🔧 Stack

| Layer | Tool |
|-------|------|
| Model | Scikit-learn + XGBoost |
| Experiment Tracking | MLflow |
| Data Versioning | DVC + Google Cloud Storage |
| API | FastAPI + Uvicorn |
| Validation | Pydantic |
| Tests | Pytest |
| Container | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Registry | Docker Hub |

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker Desktop
- DVC

### 1. Clone the repository
```bash
git clone https://github.com/JrCapela/churn-prediction.git
cd churn-prediction
```

### 2. Install dependencies
```bash
make install-dev
```

### 3. Pull data and models from GCS
```bash
make dvc-pull
```

### 4. Run the API
```bash
make run
```

API available at `http://localhost:8000`
Swagger UI at `http://localhost:8000/docs`

### 5. Run with Docker
```bash
make docker-run
```

## 🧪 Tests

```bash
# Unit tests only (no data/models needed)
make test-unit

# Integration tests (requires data and models)
make test-integration

# All tests
make test
```

## 📈 Model Performance

| Model | AUC-ROC |
|-------|---------|
| Logistic Regression | **0.839** 🥇 |
| XGBoost | 0.826 |
| Random Forest | 0.818 |

## 🔍 Key Insights (SHAP)

Top features driving churn:
- `tenure` — longer customers = less churn 🔵
- `InternetService_Fiber optic` — expensive service = more churn 🔴
- `Contract_Month-to-month` — no commitment = more churn 🔴
- `TechSupport` — supported customers stay 🔵

## 🌐 API Usage

### Predict churn for a customer

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Response
```json
{
  "churn_probability": 0.8939,
  "churn_prediction": true,
  "risk_level": "High",
  "message": "High churn risk! Immediate retention action recommended."
}
```

## ⚙️ CI/CD Pipeline
Push to main
↓
CI Pipeline (GitHub Actions)
→ Install dependencies
→ DVC pull from GCS
→ Run unit tests
→ Run integration tests
↓
CD Pipeline (GitHub Actions)
→ DVC pull from GCS
→ Build Docker image
→ Push to Docker Hub

## 🐳 Docker Hub

```bash
docker pull capelajr/churn-prediction-api:latest
```

## 📋 Makefile Commands

| Command | Description |
|---------|-------------|
| `make install` | Install production dependencies |
| `make install-dev` | Install dev dependencies |
| `make train` | Train the model |
| `make test` | Run all tests |
| `make test-unit` | Run unit tests only |
| `make test-integration` | Run integration tests only |
| `make run` | Start the API locally |
| `make docker-run` | Start with Docker |
| `make dvc-push` | Push data/models to GCS |
| `make dvc-pull` | Pull data/models from GCS |