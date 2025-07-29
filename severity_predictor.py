from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import zipfile
import os
import numpy as np
import pandas as pd

app = FastAPI()

# Unzip and load model files
model_path = "final_model.pkl"
if not os.path.exists(model_path):
    with zipfile.ZipFile("final_model.zip", "r") as zip_ref:
        zip_ref.extractall(".")

try:
    model = joblib.load(model_path)
    preprocessor = joblib.load("model_preprocessor.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load model files: {e}")

# Define request schema
class EventInput(BaseModel):
    sub_event_type: str
    disorder_type: str
    primary_actor: str
    secondary_actor: str
    interaction: str
    admin1: str
    admin3: str
    location: str
    year: int
    time_precision: int
    latitude: float
    longitude: float
    month: int

@app.post("/predict")
async def predict(event: EventInput):
    try:
        input_df = pd.DataFrame([event.dict()])
        transformed = preprocessor.transform(input_df)
        probs = model.predict_proba(transformed)
        critical_index = list(label_encoder.classes_).index("Critical")
        critical_prob = probs[0][critical_index]
        severity = (
            "Critical" if critical_prob > 0.3
            else label_encoder.inverse_transform([np.argmax(probs)])[0]
        )
        return {
            "predicted_severity": severity,
         
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
