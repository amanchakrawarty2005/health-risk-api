import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import mlflow
import os

def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Diabetes Risk Model")
    plt.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs("notebooks", exist_ok=True)
    plt.savefig("notebooks/roc_curve.png")
    plt.close()
    print(f"ROC curve saved! AUC: {roc_auc:.3f}")
    return roc_auc

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Diabetes", "Diabetes"]
    )
    plt.figure(figsize=(8, 6))
    disp.plot(cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix - Diabetes Risk Model")
    plt.tight_layout()
    plt.savefig("notebooks/confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved!")

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    feat_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x="importance", y="feature", data=feat_df, palette="viridis")
    plt.title("Feature Importance - XGBoost")
    plt.tight_layout()
    plt.savefig("notebooks/feature_importance.png")
    plt.close()
    print("Feature importance plot saved!")
    return feat_df

def run_evaluation(model, X_test, y_test, y_pred, y_prob):
    print("\n=== RUNNING FULL EVALUATION ===")

    plot_roc_curve(y_test, y_prob)
    plot_confusion_matrix(y_test, y_pred)
    feat_df = plot_feature_importance(model, X_test.columns.tolist())

    print("\n=== TOP 5 MOST IMPORTANT FEATURES ===")
    print(feat_df.head())

    mlflow.log_artifact("notebooks/roc_curve.png")
    mlflow.log_artifact("notebooks/confusion_matrix.png")
    mlflow.log_artifact("notebooks/feature_importance.png")

    print("\n✅ All evaluation plots logged to MLflow!")