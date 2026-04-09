import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop customerID — just an identifier
    df.drop(columns=['customerID'], inplace=True)

    # Fix TotalCharges — convert to float, fill nulls with median
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Drop TotalCharges — highly correlated with tenure (0.83)
    df.drop(columns=['TotalCharges'], inplace=True)

    # Encode target
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Binary columns — Yes/No → 1/0
    binary_cols = [
        'Partner', 'Dependents', 'PhoneService',
        'PaperlessBilling', 'TechSupport', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'StreamingTV', 'StreamingMovies'
    ]
    for col in binary_cols:
        df[col] = (df[col] == 'Yes').astype(int)

    # Gender — Male/Female → 1/0
    df['gender'] = (df['gender'] == 'Male').astype(int)

    # MultipleLines — has 3 values including 'No phone service'
    df['MultipleLines'] = (df['MultipleLines'] == 'Yes').astype(int)

    # One-hot encoding for categorical columns
    cat_cols = ['InternetService', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    return df


def scale_features(df: pd.DataFrame, scaler=None, fit: bool = True):
    df = df.copy()

    num_cols = ['tenure', 'MonthlyCharges']

    if fit:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])

    return df, scaler


def split_data(df: pd.DataFrame, target: str = 'Churn', test_size: float = 0.2):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


def preprocess_pipeline(path: str, save_scaler: bool = True):
    df = load_data(path)
    df = clean_data(df)
    df = encode_features(df)
    df, scaler = scale_features(df, fit=True)

    X_train, X_test, y_train, y_test = split_data(df)

    if save_scaler:
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/scaler.pkl')
        print("Scaler saved to models/scaler.pkl ✅")

    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"Churn rate train: {y_train.mean():.2%}")
    print(f"Churn rate test:  {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test