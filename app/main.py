from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Union
import torch
import numpy as np

from .model import GenomicsTabularNN
from .encoding import process_tabular_data

# requests model for prediction
class TabularInput(BaseModel):
    data: Dict[str, float]  # Contains A0-A179 as keys
    metadata: dict = None

app = FastAPI(
    title="Genetics Tabular Classifier",
    description="API for classifying genetic splice-junction tabular data",
    version="2.0.0"
)

# initializing model
model = GenomicsTabularNN(input_size=180, num_classes=4)
model.eval()

# CORS middleware implementation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message": "Genomics Classifier (Splice-junction dataset) API",
        "endpoints": {
            "POST /predict": "Make prediction on Splice-junction data",
            "GET /health": "Check API status"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_ready": True,
        "device": next(model.parameters()).device.type
    }

@app.post("/predict")
async def predict(input_data: TabularInput):
    try:
        # process input
        input_tensor = process_tabular_data(input_data.data)
        
        # make prediction
        with torch.no_grad():
            probs = model.predict_proba(input_tensor)
            confidence, prediction = torch.max(probs, dim=1)
            
        return {
            "prediction": int(prediction.item()),
            "confidence": float(confidence.item()),
            "probabilities": probs[0].tolist(),
            "metadata": input_data.metadata or {}
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)