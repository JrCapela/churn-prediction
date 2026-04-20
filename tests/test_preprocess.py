import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import clean_data, encode_features, preprocess_pipeline


def get_sample_df():
    return pd.DataFrame([{
        'customerID': '1234-ABCD',
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 85.0,
        'TotalCharges': '1020.0',
        'Churn': 'No'
    }])


@pytest.mark.unit
def test_clean_data_drops_customer_id():
    df = get_sample_df()
    result = clean_data(df)
    assert 'customerID' not in result.columns


@pytest.mark.unit
def test_clean_data_drops_total_charges():
    df = get_sample_df()
    result = clean_data(df)
    assert 'TotalCharges' not in result.columns


@pytest.mark.unit
def test_clean_data_encodes_churn():
    df = get_sample_df()
    result = clean_data(df)
    assert result['Churn'].dtype in [int, np.int64, np.int32]


@pytest.mark.unit
def test_encode_features_no_string_columns():
    df = get_sample_df()
    df = clean_data(df)
    df = encode_features(df)
    string_cols = df.select_dtypes(include='object').columns.tolist()
    assert len(string_cols) == 0


@pytest.mark.unit
def test_encode_features_creates_dummies():
    df = get_sample_df()
    df = clean_data(df)
    df = encode_features(df)
    assert 'Contract_Month-to-month' in df.columns
    assert 'InternetService_Fiber optic' in df.columns


@pytest.mark.integration
def test_preprocess_pipeline_shape():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_dir, 'data', 'raw', 'telco_churn.csv')
    X_train, X_test, y_train, y_test = preprocess_pipeline(data_path, save_scaler=False)
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train.shape[1] == X_test.shape[1]