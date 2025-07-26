#!/bin/bash
gunicorn -w 1 -k uvicorn.workers.UvicornWorker severity_predictor:app --bind 0.0.0.0:8000


