import sqlite3
import os
from datetime import datetime

# ── We use SQLite for now (no setup needed) ───────────────────
# In production this would be PostgreSQL
DB_PATH = "predictions.db"

# ── Create Table ──────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL,
            client_name TEXT NOT NULL,
            risk_score REAL NOT NULL,
            risk_label TEXT NOT NULL,
            model_version TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("Database initialized!")

# ── Save Prediction ───────────────────────────────────────────
def save_prediction(request_id, client_name, risk_score, risk_label, model_version):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions
        (request_id, client_name, risk_score, risk_label, model_version, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        request_id,
        client_name,
        risk_score,
        risk_label,
        model_version,
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()

# ── Get Usage Stats ───────────────────────────────────────────
def get_usage(client_name: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) as total_requests,
               AVG(risk_score) as avg_risk_score,
               MAX(timestamp) as last_request
        FROM predictions
        WHERE client_name = ?
    """, (client_name,))
    row = cursor.fetchone()
    conn.close()
    return {
        "total_requests": row[0],
        "avg_risk_score": round(row[1], 3) if row[1] else 0,
        "last_request": row[2]
    }