from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import json
from datetime import datetime

app = FastAPI()

model = joblib.load("models/best_model.pkl")

class Input(BaseModel):
    track_count: int = Field(..., ge=2, le=100)
    sample_rate_khz: int = Field(..., ge=22, le=500)
    dynamic_range_db: float = Field(..., ge=6, le=20)
    is_mastered: int = Field(..., ge=0, le=1)

@app.get("/ping")
def ping():
    return {"status": "healthy", "model_loaded": True}

@app.post("/infer")
def infer(data: Input):
    pred = model.predict([[data.track_count, data.sample_rate_khz,
                           data.dynamic_range_db, data.is_mastered]])[0]

    log = {
        "timestamp": str(datetime.now()),
        "input": data.dict(),
        "prediction": float(pred)
    }

    with open("logs/predictions.jsonl", "a") as f:
        f.write(json.dumps(log) + "\n")

    return {"prediction": float(pred)}
