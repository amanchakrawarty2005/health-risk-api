import streamlit as st
import requests
import plotly.express as px
import pandas as pd

API_URL = "http://localhost:8000"
API_KEY = "test-key-123"

st.set_page_config(
    page_title="Health Risk Prediction Dashboard",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Health Risk Prediction Dashboard")
st.markdown("B2B Diabetes Risk Prediction API — Powered by XGBoost + CDC BRFSS Data")

st.sidebar.header("Patient Information")

HighBP = st.sidebar.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
HighChol = st.sidebar.selectbox("High Cholesterol", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
CholCheck = st.sidebar.selectbox("Cholesterol Check", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
BMI = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
Smoker = st.sidebar.selectbox("Smoker", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
Stroke = st.sidebar.selectbox("Stroke History", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
HeartDiseaseorAttack = st.sidebar.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
PhysActivity = st.sidebar.selectbox("Physical Activity", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
Fruits = st.sidebar.selectbox("Eats Fruits Daily", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
Veggies = st.sidebar.selectbox("Eats Veggies Daily", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
HvyAlcoholConsump = st.sidebar.selectbox("Heavy Alcohol", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
AnyHealthcare = st.sidebar.selectbox("Has Healthcare", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
NoDocbcCost = st.sidebar.selectbox("No Doctor due to Cost", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
GenHlth = st.sidebar.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)
MentHlth = st.sidebar.slider("Poor Mental Health Days (0-30)", 0, 30, 0)
PhysHlth = st.sidebar.slider("Poor Physical Health Days (0-30)", 0, 30, 0)
DiffWalk = st.sidebar.selectbox("Difficulty Walking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
Sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
Age = st.sidebar.slider("Age Group (1=18-24, 13=80+)", 1, 13, 7)
Education = st.sidebar.slider("Education Level (1-6)", 1, 6, 4)
Income = st.sidebar.slider("Income Level (1-8)", 1, 8, 4)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("BMI", BMI)
with col2:
    st.metric("Age Group", Age)
with col3:
    st.metric("General Health", GenHlth)

st.divider()

if st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True):
    payload = {
        "HighBP": HighBP, "HighChol": HighChol, "CholCheck": CholCheck,
        "BMI": BMI, "Smoker": Smoker, "Stroke": Stroke,
        "HeartDiseaseorAttack": HeartDiseaseorAttack, "PhysActivity": PhysActivity,
        "Fruits": Fruits, "Veggies": Veggies, "HvyAlcoholConsump": HvyAlcoholConsump,
        "AnyHealthcare": AnyHealthcare, "NoDocbcCost": NoDocbcCost,
        "GenHlth": GenHlth, "MentHlth": MentHlth, "PhysHlth": PhysHlth,
        "DiffWalk": DiffWalk, "Sex": Sex, "Age": Age,
        "Education": Education, "Income": Income
    }

    with st.spinner("Analyzing patient data..."):
        try:
            response = requests.post(
                f"{API_URL}/v1/predict",
                headers={"X-API-Key": API_KEY},
                json=payload
            )
            result = response.json()

            st.divider()
            col1, col2, col3 = st.columns(3)

            with col1:
                score = result["risk_score"]
                color = "red" if score >= 0.7 else "orange" if score >= 0.4 else "green"
                st.markdown(f"### Risk Score")
                st.markdown(f"<h1 style='color:{color}'>{score:.1%}</h1>", unsafe_allow_html=True)

            with col2:
                st.markdown(f"### Risk Label")
                label = result["risk_label"]
                icon = "🔴" if label == "High Risk" else "🟠" if label == "Medium Risk" else "🟢"
                st.markdown(f"<h1>{icon} {label}</h1>", unsafe_allow_html=True)

            with col3:
                st.markdown(f"### Confidence")
                st.markdown(f"<h1>{result['confidence']:.1%}</h1>", unsafe_allow_html=True)

            st.divider()
            st.subheader("Top Risk Factors")

            factors = result["top_risk_factors"]
            df_factors = pd.DataFrame(factors)

            fig = px.bar(
                df_factors,
                x="impact",
                y="feature",
                orientation="h",
                color="impact",
                color_continuous_scale="RdYlGn_r",
                title="SHAP Feature Impact on Risk Score"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("API Response Details")
            st.json(result)

        except Exception as e:
            st.error(f"API Error: {e}. Make sure the API is running at {API_URL}")

st.divider()
st.subheader("Model Information")

try:
    health = requests.get(f"{API_URL}/v1/health").json()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("API Status", health["status"])
    with col2:
        st.metric("Model Version", health["version"])
    with col3:
        st.metric("Model Loaded", "Yes" if health["model_loaded"] else "No")
except:
    st.warning("API not reachable. Start the API first.")