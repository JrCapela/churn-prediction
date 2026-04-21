import os
import sys
import optuna
import mlflow
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess_pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_data():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_dir, 'data', 'raw', 'telco_churn.csv')
    return preprocess_pipeline(data_path, save_scaler=True)


def objective_logistic(trial, X_train, X_test, y_train, y_test):
    params = {
        "C": trial.suggest_float("C", 0.01, 10.0, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 1000),
        "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
        "class_weight": "balanced",
        "random_state": 42
    }
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


def objective_xgboost(trial, X_train, X_test, y_train, y_test):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": 3,
        "random_state": 42,
        "eval_metric": "logloss"
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


def objective_random_forest(trial, X_train, X_test, y_train, y_test):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "class_weight": "balanced",
        "random_state": 42
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


def tune_all(n_trials: int = 30):
    print("Loading data...")
    X_train, X_test, y_train, y_test = get_data()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("churn-prediction-tuning")

    objectives = {
        "LogisticRegression": objective_logistic,
        "XGBoost": objective_xgboost,
        "RandomForest": objective_random_forest
    }

    best_results = {}

    for name, objective in objectives.items():
        print(f"\nTuning {name} with {n_trials} trials...")

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, X_train, X_test, y_train, y_test),
            n_trials=n_trials,
            show_progress_bar=True
        )

        best_auc = study.best_value
        best_params = study.best_params
        print(f"{name} — Best AUC-ROC: {best_auc:.4f}")
        print(f"Best params: {best_params}")

        # Log no MLflow
        with mlflow.start_run(run_name=f"{name}_tuned"):
            mlflow.log_params(best_params)
            mlflow.log_metric("auc_roc_tuned", best_auc)

        best_results[name] = {
            "auc": best_auc,
            "params": best_params
        }

    # Encontrar o melhor modelo
    best_model_name = max(best_results, key=lambda x: best_results[x]["auc"])
    print(f"\n🏆 Best model: {best_model_name} — AUC: {best_results[best_model_name]['auc']:.4f}")

    return best_results


if __name__ == "__main__":
    results = tune_all(n_trials=30)