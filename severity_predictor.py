# FastAPI Setup for Serving
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Load model artifacts
model = joblib.load('voting_severity_model.pkl')
preprocessor = joblib.load('model_preprocessor.pkl')
label_encoder = joblib.load('label_encoder.pkl')

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

# Prediction endpoint
@app.post("/predict")
def predict(event: EventInput):
    df = pd.DataFrame([event.dict()])
    transformed = preprocessor.transform(df)
    probs = model.predict_proba(transformed)
    
    critical_prob = probs[0] [list(label_encoder.classes_).index("Critical")]
    severity = "Critical" if critical_prob > 0.3 else label_encoder.inverse_transform([np.argmax(probs)])[0]
    
    return {
        "predicted_severity": severity,
        "critical_probability": round(float(critical_prob), 3)
    }
