import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load model and columns at startup
with open("body_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("input_cols.pkl", "rb") as f:
    input_cols = pickle.load(f)

app = FastAPI()

class BodyInput(BaseModel):
    height_cm: float
    weight_kg: float
    thigh_circum_cm: float
    hip_circum_cm: float
    waist_circum_cm: float
    chest_circum_cm: float
    bicep_circum_cm: float
    gym_days_per_week: int
    calories_per_day: float
    protein_g_per_day: float
    sleep_hours_per_day: float
    job_activity_level: int

@app.post("/predict")
def predict(body: BodyInput):
    df = pd.DataFrame([body.dict()])[input_cols]
    pred = model.predict(df)[0]

    return {
        "weight_kg": float(pred[0]),
        "thigh_circum_cm": float(pred[1]),
        "hip_circum_cm": float(pred[2]),
        "waist_circum_cm": float(pred[3]),
        "chest_circum_cm": float(pred[4]),
        "bicep_circum_cm": float(pred[5]),
        "calf_circum_cm": float(pred[6]),
    }
