import mlflow.sklearn
import mlflow
import pandas as pd
import numpy as np
import shap
import os
from dotenv import load_dotenv

load_dotenv()

# ── Model Config ──────────────────────────────────────────────
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.2.0")
FEATURE_NAMES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits",
    "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age",
    "Education", "Income"
]

# ── Load Model ────────────────────────────────────────────────
def load_model():
    try:
        import joblib
        model = joblib.load("models/xgboost_model.pkl")
        print("Model loaded from file!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# ── Global Model Instance ─────────────────────────────────────
model = load_model()

# ── Get Risk Label ────────────────────────────────────────────
def get_risk_label(score: float) -> str:
    if score >= 0.7:
        return "High Risk"
    elif score >= 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

# ── Get SHAP Explanations ─────────────────────────────────────
def get_shap_explanation(input_df: pd.DataFrame):
    try:
        booster = model.get_booster()
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(input_df)

        # Get top 3 risk factors
        feature_impacts = []
        for i, feature in enumerate(FEATURE_NAMES):
            feature_impacts.append({
                "feature": feature,
                "value": float(input_df.iloc[0][feature]),
                "impact": float(shap_values[0][i])
            })

        # Sort by absolute impact
        feature_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return feature_impacts[:3]

    except Exception as e:
        print(f"SHAP error: {e}")
        return []

# ── Main Predict Function ─────────────────────────────────────
def predict(data: dict):
    # Convert to DataFrame
    input_df = pd.DataFrame([data], columns=FEATURE_NAMES)

    # Get prediction
    risk_score = float(model.predict_proba(input_df)[0][1])
    risk_label = get_risk_label(risk_score)
    confidence = round(max(
        model.predict_proba(input_df)[0]
    ), 3)

    # Get SHAP explanations
    top_risk_factors = get_shap_explanation(input_df)

    return {
        "risk_score": round(risk_score, 3),
        "risk_label": risk_label,
        "confidence": confidence,
        "top_risk_factors": top_risk_factors,
        "model_version": MODEL_VERSION
    }