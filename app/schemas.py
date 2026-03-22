from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# ── Input Schema ──────────────────────────────────────────────
class PredictRequest(BaseModel):
    HighBP: float = Field(..., ge=0, le=1, description="High Blood Pressure (0 or 1)")
    HighChol: float = Field(..., ge=0, le=1, description="High Cholesterol (0 or 1)")
    CholCheck: float = Field(..., ge=0, le=1, description="Cholesterol Check (0 or 1)")
    BMI: float = Field(..., ge=10, le=100, description="Body Mass Index")
    Smoker: float = Field(..., ge=0, le=1, description="Smoker (0 or 1)")
    Stroke: float = Field(..., ge=0, le=1, description="Stroke history (0 or 1)")
    HeartDiseaseorAttack: float = Field(..., ge=0, le=1, description="Heart Disease (0 or 1)")
    PhysActivity: float = Field(..., ge=0, le=1, description="Physical Activity (0 or 1)")
    Fruits: float = Field(..., ge=0, le=1, description="Fruits consumption (0 or 1)")
    Veggies: float = Field(..., ge=0, le=1, description="Vegetables consumption (0 or 1)")
    HvyAlcoholConsump: float = Field(..., ge=0, le=1, description="Heavy Alcohol (0 or 1)")
    AnyHealthcare: float = Field(..., ge=0, le=1, description="Any Healthcare (0 or 1)")
    NoDocbcCost: float = Field(..., ge=0, le=1, description="No Doctor due to cost (0 or 1)")
    GenHlth: float = Field(..., ge=1, le=5, description="General Health (1-5)")
    MentHlth: float = Field(..., ge=0, le=30, description="Mental Health days (0-30)")
    PhysHlth: float = Field(..., ge=0, le=30, description="Physical Health days (0-30)")
    DiffWalk: float = Field(..., ge=0, le=1, description="Difficulty Walking (0 or 1)")
    Sex: float = Field(..., ge=0, le=1, description="Sex (0=Female, 1=Male)")
    Age: float = Field(..., ge=1, le=13, description="Age Group (1-13)")
    Education: float = Field(..., ge=1, le=6, description="Education Level (1-6)")
    Income: float = Field(..., ge=1, le=8, description="Income Level (1-8)")

# ── Risk Factor Schema ─────────────────────────────────────────
class RiskFactor(BaseModel):
    feature: str
    value: float
    impact: float

# ── Output Schema ─────────────────────────────────────────────
class PredictResponse(BaseModel):
    request_id: str
    risk_score: float
    risk_label: str
    confidence: float
    top_risk_factors: List[RiskFactor]
    model_version: str
    timestamp: datetime

# ── Health Check Schema ───────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    timestamp: datetime