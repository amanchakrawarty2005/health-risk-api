import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

def setup_mlflow(experiment_name="health-risk-v1"):
    mlflow.set_experiment(experiment_name)
    print(f"MLflow experiment set: {experiment_name}")

def log_params(params: dict):
    for key, value in params.items():
        mlflow.log_param(key, value)

def log_metrics(metrics: dict):
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

def log_model(model, model_name="xgboost_model"):
    mlflow.sklearn.log_model(model, model_name)
    print(f"Model logged: {model_name}")

def log_artifact(filepath):
    mlflow.log_artifact(filepath)
    print(f"Artifact logged: {filepath}")