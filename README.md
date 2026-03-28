# 🏥 Health Risk Prediction API

A production-ready B2B Machine Learning API that predicts diabetes risk using the CDC BRFSS dataset (253,680 records).

🔴 **Live API:** https://health-risk-api-3fw9.onrender.com/docs  
🟢 **Live Dashboard:** https://health-risk-api.streamlit.app

---

## 🚀 Quick Demo

```bash
curl -X POST https://health-risk-api-3fw9.onrender.com/v1/predict \
  -H "X-API-Key: test-key-123" \
  -H "Content-Type: application/json" \
  -d '{
    "HighBP": 1, "HighChol": 1, "CholCheck": 1,
    "BMI": 34.2, "Smoker": 1, "Stroke": 0,
    "HeartDiseaseorAttack": 0, "PhysActivity": 0,
    "Fruits": 0, "Veggies": 0, "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 4,
    "MentHlth": 10, "PhysHlth": 10, "DiffWalk": 1,
    "Sex": 1, "Age": 9, "Education": 4, "Income": 3
  }'
```

**Response:**
```json
{
  "request_id": "uuid-here",
  "risk_score": 0.881,
  "risk_label": "High Risk",
  "confidence": 0.88,
  "top_risk_factors": [
    {"feature": "GenHlth", "value": 4, "impact": 0.496},
    {"feature": "BMI", "value": 34.2, "impact": 0.438},
    {"feature": "HighBP", "value": 1, "impact": 0.409}
  ],
  "model_version": "v1.2.0",
  "timestamp": "2026-03-28T10:00:00Z"
}
```

---

## 📊 EDA Insights

### Class Distribution
![Class Distribution](images/no%20diabetes%20vs%20diabetes%20bar%20graph.png)

### BMI Distribution by Diabetes Status
![BMI Distribution](images/bmi_distribution.png)

### Feature Correlation Heatmap
![Correlation Heatmap](images/correlation_heatmap.png)

### Risk Factor Comparison
![Risk Factors](images/risk_factors.png)

### Diabetes Cases by Age Group
![Age Distribution](images/age_distribution.png)

---

## 🤖 Model Performance

| Metric | Score |
|--------|-------|
| AUC | **0.825** |
| Accuracy | 72.5% |
| Recall | 78.1% |
| F1 Score | 0.44 |

### ROC Curve
![ROC Curve](images/roc_curve.png)

### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

### Feature Importance
![Feature Importance](images/feature_importance.png)

### SHAP Summary
![SHAP Summary](images/shap_summary.png)

---

## 🏗️ Architecture

```
Client (Hospital/App)
        ↓
   API Key Auth        ← X-API-Key header
        ↓
   FastAPI Backend     ← /v1/predict
        ↓
   XGBoost Model       ← AUC: 0.825
        ↓
   SHAP Explanation    ← Top 3 risk factors
        ↓
   SQLite Logging      ← Every prediction saved
        ↓
   JSON Response       ← risk_score + risk_label
```

---

## 📁 Project Structure

```
health-risk-api/
├── app/
│   ├── main.py           ← FastAPI entry point
│   ├── auth.py           ← API key authentication
│   ├── schemas.py        ← Pydantic request/response models
│   ├── predict.py        ← Model loading + SHAP
│   └── database.py       ← SQLite logging
├── training/
│   ├── train.py          ← XGBoost training pipeline
│   ├── evaluate.py       ← ROC, confusion matrix, feature importance
│   ├── mlflow_logger.py  ← Experiment tracking
│   └── drift_detection.py← Evidently drift monitoring
├── dashboard/
│   └── streamlit_app.py  ← Live frontend dashboard
├── models/
│   └── xgboost_model.pkl ← Trained model
├── notebooks/
│   └── eda.ipynb         ← Exploratory data analysis
├── images/               ← All EDA + evaluation plots
├── data/                 ← CDC BRFSS dataset
├── tests/
│   └── test_api.py       ← API tests
├── .github/
│   └── workflows/
│       └── deploy.yml    ← CI/CD pipeline
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| XGBoost | ML model (AUC: 0.825) |
| FastAPI | REST API |
| MLflow | Experiment tracking |
| SHAP | Model explainability |
| Evidently | Data drift detection |
| Docker | Containerization |
| GitHub Actions | CI/CD pipeline |
| Streamlit | Live dashboard |
| SQLite | Prediction logging |
| Render | Cloud deployment |

---

## ✨ Key Features

- **B2B Authentication** — API key based access control
- **Versioned endpoints** — `/v1/predict`, `/v1/health`, `/v1/usage`
- **SHAP explanations** — every prediction explains the top 3 risk factors
- **Data drift detection** — Evidently AI monitors model health
- **CI/CD pipeline** — auto-tests + Docker build on every git push
- **Live dashboard** — real-time predictions via Streamlit

---

## 🚀 Local Setup

```bash
git clone https://github.com/amanchakrawarty2005/health-risk-api.git
cd health-risk-api
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python training/train.py
uvicorn app.main:app --reload
```

Then open: http://localhost:8000/docs

---

## 📈 MLflow Experiment Tracking

```bash
mlflow ui
```

Open: http://localhost:5000

---

## 🐳 Docker

```bash
docker build -t health-risk-api .
docker run -p 8000:8000 health-risk-api
```

---

## 📝 Resume Bullet Points

- Built B2B diabetes risk prediction API using XGBoost trained on 253K CDC BRFSS records, deployed on Render with Docker and CI/CD via GitHub Actions
- Implemented SHAP explainability returning top 3 risk factors per prediction, and automated data drift detection using Evidently AI
- Tracked 15+ ML experiments using MLflow, achieving AUC of 0.825 with automated model selection pipeline
- Built live Streamlit dashboard at health-risk-api.streamlit.app with real-time predictions and SHAP visualizations

---

## 👤 Author

**Aman Chakrawarty**
- GitHub: [@amanchakrawarty2005](https://github.com/amanchakrawarty2005)
- Live API: https://health-risk-api-3fw9.onrender.com/docs
- Live Dashboard: https://health-risk-api.streamlit.app