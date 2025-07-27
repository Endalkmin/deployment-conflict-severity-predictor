# FastAPI Setup for Serving
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Input data structure
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

# Load model artifacts safely
try:
    model = joblib.load("voting_severity_model.pkl")
    preprocessor = joblib.load("model_preprocessor.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except Exception as e:
    print(f"[STARTUP ERROR] Failed to load model files: {e}")
    raise RuntimeError("App failed to load required artifacts")

@app.post("/predict")
async def predict(event: EventInput):
    try:
        input_df = pd.DataFrame([event.dict()])
        print(f"[INPUT DF]:\n{input_df}")

        transformed = preprocessor.transform(input_df)
        print(f"[TRANSFORMED SHAPE]: {transformed.shape}")

        probs = model.predict_proba(transformed)
        print(f"[PROBS]: {probs}")

        critical_index = list(label_encoder.classes_).index("Critical")
        critical_prob = probs[0][critical_index]
        print(f"[CRITICAL PROB]: {critical_prob}")

        severity = (
            "Critical" if critical_prob > 0.3 
            else label_encoder.inverse_transform([np.argmax(probs)])[0]
        )
        print(f"[PREDICTED SEVERITY]: {severity}")

        return {
            "predicted_severity": severity,
            "critical_probability": round(float(critical_prob), 3)
        }

    except Exception as e:
        print(f"[ERROR TRACE]: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
