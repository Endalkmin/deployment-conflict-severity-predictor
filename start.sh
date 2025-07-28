#!/bin/bash

# Unzip model if needed
if [ -f "voting_model.zip" ]; then
    echo "[INFO] Extracting model from voting_model.zip..."
    unzip -o tuned_voting_model.zip -d .
fi

# Launch FastAPI app with Gunicorn
exec gunicorn severity_predictor:app --workers 1 --bind 0.0.0.0:8000

