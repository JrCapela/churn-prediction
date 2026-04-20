import pytest
import joblib
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess_pipeline


def get_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.integration
def test_model_file_exists():
    root = get_root()
    assert os.path.exists(os.path.join(root, 'models', 'LogisticRegression.pkl'))


@pytest.mark.integration
def test_scaler_file_exists():
    root = get_root()
    assert os.path.exists(os.path.join(root, 'models', 'scaler.pkl'))


@pytest.mark.integration
def test_model_predicts_probability():
    root = get_root()
    model = joblib.load(os.path.join(root, 'models', 'LogisticRegression.pkl'))
    feature_columns = joblib.load(os.path.join(root, 'models', 'feature_columns.pkl'))
    data_path = os.path.join(root, 'data', 'raw', 'telco_churn.csv')
    X_train, X_test, y_train, y_test = preprocess_pipeline(data_path, save_scaler=False)
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)
    probas = model.predict_proba(X_test)
    assert probas.shape[1] == 2
    assert np.all(probas >= 0) and np.all(probas <= 1)


@pytest.mark.integration
def test_model_output_range():
    root = get_root()
    model = joblib.load(os.path.join(root, 'models', 'LogisticRegression.pkl'))
    feature_columns = joblib.load(os.path.join(root, 'models', 'feature_columns.pkl'))
    data_path = os.path.join(root, 'data', 'raw', 'telco_churn.csv')
    X_train, X_test, y_train, y_test = preprocess_pipeline(data_path, save_scaler=False)
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)
    predictions = model.predict(X_test)
    assert set(predictions).issubset({0, 1})