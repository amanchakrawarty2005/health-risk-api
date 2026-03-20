import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import shap
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.mlflow_logger import setup_mlflow, log_params, log_metrics, log_model, log_artifact

# ── 1. Load Data ──────────────────────────────────────────────
def load_data():
    df = pd.read_csv("data/diabetes_binary_health_indicators_BRFSS2015.csv")
    print(f"Dataset loaded: {df.shape}")
    return df

# ── 2. Preprocess ─────────────────────────────────────────────
def preprocess(df):
    X = df.drop("Diabetes_binary", axis=1)
    y = df["Diabetes_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# ── 3. Train Model ────────────────────────────────────────────
def train_model(X_train, y_train):
    # Handle class imbalance
    scale = (y_train == 0).sum() / (y_train == 1).sum()

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": scale,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    print("Model trained successfully!")
    return model, params

# ── 4. Evaluate ───────────────────────────────────────────────
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "auc":       round(roc_auc_score(y_test, y_prob), 4),
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
    }

    print("\n=== MODEL METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return metrics, y_pred, y_prob

# ── 5. SHAP Plot ──────────────────────────────────────────────
def generate_shap(model, X_test):
    os.makedirs("notebooks", exist_ok=True)
    print("Generating SHAP values...")

    booster = model.get_booster()
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_test[:500])

    plt.figure()
    shap.summary_plot(shap_values, X_test[:500], show=False)
    plt.tight_layout()
    plt.savefig("notebooks/shap_summary.png")
    plt.close()
    print("SHAP plot saved!")

# ── 6. Main ───────────────────────────────────────────────────
def main():
    setup_mlflow("health-risk-v1")

    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)

    with mlflow.start_run():
        model, params = train_model(X_train, y_train)
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)

        # Log everything to MLflow
        log_params(params)
        log_metrics(metrics)
        log_model(model, "xgboost_model")

        # SHAP
        generate_shap(model, X_test)
        log_artifact("notebooks/shap_summary.png")

        print("\n✅ Run logged to MLflow!")
        print(f"AUC: {metrics['auc']} | F1: {metrics['f1']}")

if __name__ == "__main__":
    main()