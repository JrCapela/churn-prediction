import os
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.preprocess import preprocess_pipeline


def get_models():
    return {
        "LogisticRegression": LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            class_weight='balanced',
            n_estimators=100,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=3,
            n_estimators=100,
            random_state=42,
            eval_metric='logloss'
        )
    }


def train_all(data_path: str):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("churn-prediction")

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    absolute_path = os.path.join(root_dir, data_path) if not os.path.isabs(data_path) else data_path
    X_train, X_test, y_train, y_test = preprocess_pipeline(absolute_path, save_scaler=True)

    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)

            # Log parameters
            mlflow.log_params(model.get_params())

            # Log model
            mlflow.sklearn.log_model(model, artifact_path="model")

            # Save locally
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, f"models/{name}.pkl")
            print(f"{name} saved ✅")

            results[name] = model

    return results, X_test, y_test


if __name__ == "__main__":
    train_all("data/raw/telco_churn.csv")