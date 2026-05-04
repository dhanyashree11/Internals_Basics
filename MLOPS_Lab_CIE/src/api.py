from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import datetime
import json
import os

app = FastAPI()
model = joblib.load("models/best_model.pkl")

class InputData(BaseModel):
    track_count: int = Field(..., ge=2, le=100)
    sample_rate_khz: int = Field(..., ge=20, le=400)
    dynamic_range_db: float = Field(..., ge=5, le=25)
    is_mastered: int = Field(..., ge=0, le=1)

@app.get("/ping")
def ping():
    return {"status": "healthy", "model_loaded": True}

@app.post("/infer")
def infer(data: InputData):
    features = np.array([[data.track_count, data.sample_rate_khz,
                          data.dynamic_range_db, data.is_mastered]])
    pred = model.predict(features)[0]

    os.makedirs("logs", exist_ok=True)
    log_entry = {
        "timestamp": str(datetime.datetime.now()),
        "input": data.dict(),
        "prediction": float(pred)
    }

    with open("logs/predictions.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {"prediction": float(pred)}
