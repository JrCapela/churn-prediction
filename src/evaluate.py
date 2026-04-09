import numpy as np
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)

    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

    with mlflow.start_run(run_name=f"{name}_eval"):
        mlflow.log_metric("auc_roc", auc)

    return {"name": name, "auc": auc, "model": model}


def evaluate_all(results, X_test, y_test):
    metrics = []

    for name, model in results.items():
        m = evaluate_model(name, model, X_test, y_test)
        metrics.append(m)

    return metrics


def plot_roc_curves(results, X_test, y_test):
    plt.figure(figsize=(8, 6))

    for name, model in results.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves — All Models')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Churn', 'Churn'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix — {name}')
    plt.tight_layout()
    plt.show()