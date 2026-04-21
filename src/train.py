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
            C=0.115,
            max_iter=816,
            solver='saga',
            class_weight='balanced',
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=149,
            max_depth=13,
            min_samples_split=7,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=151,
            max_depth=5,
            learning_rate=0.0245,
            subsample=0.814,
            colsample_bytree=0.836,
            scale_pos_weight=3,
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

    # Save feature columns for API alignment
    columns_path = os.path.join(root_dir, 'models', 'feature_columns.pkl')
    joblib.dump(list(X_train.columns), columns_path)
    print(f"Feature columns saved ✅")

    models = get_models()
    results = {}

    for name, model in models.items():
        model_path = os.path.join(root_dir, 'models', f'{name}.pkl')
        print(f"\nTraining {name}...")
        print(f"Saving to: {model_path}")

        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            mlflow.log_params(model.get_params())
            mlflow.sklearn.log_model(model, artifact_path="model")

            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            print(f"{name} saved ✅")

            results[name] = model

    return results, X_test, y_test


if __name__ == "__main__":
    train_all("data/raw/telco_churn.csv")