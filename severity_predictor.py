from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and label encoder
model = joblib.load("voting_severity_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

app = FastAPI()

# Input schema
class EventFeatures(BaseModel):
    features: list  # must match preprocessed shape

@app.post("/predict")
def predict_severity(data: EventFeatures):
    try:
        prediction = model.predict(np.array([data.features]))[0]
        severity = label_encoder.inverse_transform([prediction])[0]
        return {"severity_level": severity}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
