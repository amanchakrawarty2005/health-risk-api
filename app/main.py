from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
from datetime import datetime
import os
from dotenv import load_dotenv

from app.schemas import PredictRequest, PredictResponse, HealthResponse, RiskFactor
from app.auth import verify_api_key
from app.database import init_db, save_prediction, get_usage
from app.predict import predict, model, MODEL_VERSION

load_dotenv()

# ── App Setup ─────────────────────────────────────────────────
app = FastAPI(
    title="Health Risk Prediction API",
    description="B2B API for diabetes risk prediction using CDC BRFSS data",
    version="1.0.0"
)

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup ───────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    init_db()
    print("✅ API started successfully!")

# ── Health Check ──────────────────────────────────────────────
@app.get("/v1/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version=MODEL_VERSION,
        timestamp=datetime.utcnow()
    )

# ── Predict ───────────────────────────────────────────────────
@app.post("/v1/predict", response_model=PredictResponse)
async def predict_risk(
    request: PredictRequest,
    client_name: str = Depends(verify_api_key)
):
    try:
        # Run prediction
        result = predict(request.dict())

        # Generate request ID
        request_id = str(uuid.uuid4())

        # Save to database
        save_prediction(
            request_id=request_id,
            client_name=client_name,
            risk_score=result["risk_score"],
            risk_label=result["risk_label"],
            model_version=MODEL_VERSION
        )

        # Build response
        return PredictResponse(
            request_id=request_id,
            risk_score=result["risk_score"],
            risk_label=result["risk_label"],
            confidence=result["confidence"],
            top_risk_factors=[
                RiskFactor(**f) for f in result["top_risk_factors"]
            ],
            model_version=MODEL_VERSION,
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Usage Stats ───────────────────────────────────────────────
@app.get("/v1/usage")
async def usage_stats(client_name: str = Depends(verify_api_key)):
    stats = get_usage(client_name)
    return {
        "client": client_name,
        "stats": stats
    }

# ── Root ──────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "Health Risk Prediction API",
        "version": MODEL_VERSION,
        "docs": "/docs",
        "health": "/v1/health"
    }