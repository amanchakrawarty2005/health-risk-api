import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric, DataDriftTable
import os
from datetime import datetime

# ── Feature Names ─────────────────────────────────────────────
FEATURE_NAMES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits",
    "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age",
    "Education", "Income"
]

# ── Load Reference Data ───────────────────────────────────────
def load_reference_data(n_samples=1000):
    df = pd.read_csv("data/diabetes_binary_health_indicators_BRFSS2015.csv")
    reference = df[FEATURE_NAMES].sample(n=n_samples, random_state=42)
    print(f"Reference data loaded: {reference.shape}")
    return reference

# ── Simulate Current Data ─────────────────────────────────────
def simulate_current_data(reference_data, drift=False):
    current = reference_data.copy()
    if drift:
        current["BMI"] = current["BMI"] * 1.3
        current["Age"] = current["Age"] + 2
        current["HighBP"] = (current["HighBP"] + 0.3).clip(0, 1)
        print("Drift simulated in BMI, Age, HighBP")
    else:
        current["BMI"] = current["BMI"] + np.random.normal(0, 0.5, len(current))
        print("No drift — small noise added")
    return current

# ── Run Drift Report ──────────────────────────────────────────
def run_drift_report(reference_data, current_data, save_path="notebooks"):
    os.makedirs(save_path, exist_ok=True)

    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable(),
    ])

    report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = f"{save_path}/drift_report_{timestamp}.html"
    report.save_html(html_path)
    print(f"Drift report saved: {html_path}")

    results = report.as_dict()
    drift_detected = results["metrics"][0]["result"]["dataset_drift"]

    return drift_detected, html_path, results

# ── Check Drift ───────────────────────────────────────────────
def check_drift(reference_data, current_data):
    print("\n=== RUNNING DRIFT DETECTION ===")

    drift_detected, html_path, results = run_drift_report(
        reference_data, current_data
    )

    drifted_features = results["metrics"][0]["result"]["number_of_drifted_columns"]
    total_features = results["metrics"][0]["result"]["number_of_columns"]

    print(f"\nDrift detected: {drift_detected}")
    print(f"Drifted features: {drifted_features}/{total_features}")
    print(f"Report saved at: {html_path}")

    if drift_detected:
        print("\n⚠️  ALERT: Data drift detected! Model retraining recommended.")
    else:
        print("\n✅ Model stable — no significant drift detected.")

    return drift_detected

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    reference_data = load_reference_data()

    print("\n--- Test 1: No Drift ---")
    current_no_drift = simulate_current_data(reference_data, drift=False)
    check_drift(reference_data, current_no_drift)

    print("\n--- Test 2: With Drift ---")
    current_with_drift = simulate_current_data(reference_data, drift=True)
    check_drift(reference_data, current_with_drift)