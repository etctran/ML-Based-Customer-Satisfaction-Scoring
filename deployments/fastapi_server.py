#!/usr/bin/env python3
"""
Simple FastAPI server for customer satisfaction predictions
Works on Windows without ZenML deployment issues
"""

import os
import sys
import pickle
from pathlib import Path
from typing import List

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from model.data_cleaning import DataCleaning, DataPreprocessStrategy

app = FastAPI(title="Customer Satisfaction Predictor", version="1.0.0")

# Global model variable
model = None

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    payment_sequential: int
    payment_installments: int
    payment_value: float
    price: float
    freight_value: float
    product_name_lenght: int
    product_description_lenght: int
    product_photos_qty: int
    product_weight_g: float
    product_length_cm: float
    product_height_cm: float
    product_width_cm: float

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_satisfaction: float
    confidence: str

def load_latest_model():
    """Load the latest trained model"""
    global model
    
    # Look for saved models in various locations
    possible_paths = [
        Path("saved_model"),
        Path("mlruns"),
        Path("model")
    ]
    
    model_file = None
    
    # Try to find a pickle file
    for path in possible_paths:
        if path.exists():
            for file in path.rglob("*.pkl"):
                model_file = file
                break
            if model_file:
                break
    
    if model_file and model_file.exists():
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded from: {model_file}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
    
    # Fallback: try to load from MLflow if available
    try:
        import mlflow
        import mlflow.lightgbm
        
        # Try to load the latest model from MLflow
        mlflow.set_tracking_uri("./mlruns")
        
        # This would need the specific run ID or model name
        # For now, just indicate MLflow is available
        print("MLflow available, but specific model loading needs run ID")
        
    except ImportError:
        print("Could not load model. Train a model first with: python run_pipeline.py")
    
    return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_latest_model()
    if not success:
        print("Starting server without loaded model. Train a model first!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Customer Satisfaction Predictor API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first!")
    
    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame([request.dict()])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Determine confidence level
        if prediction >= 4.5:
            confidence = "high"
        elif prediction >= 3.5:
            confidence = "medium"
        else:
            confidence = "low"
        
        return PredictionResponse(
            predicted_satisfaction=float(prediction),
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(requests: List[PredictionRequest]):
    """Make batch predictions"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first!")
    
    try:
        # Convert requests to DataFrame
        input_data = pd.DataFrame([req.dict() for req in requests])
        
        # Make predictions
        predictions = model.predict(input_data)
        
        results = []
        for pred in predictions:
            confidence = "high" if pred >= 4.5 else "medium" if pred >= 3.5 else "low"
            results.append({
                "predicted_satisfaction": float(pred),
                "confidence": confidence
            })
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

def main():
    """Run the FastAPI server"""
    print("Starting Customer Satisfaction Prediction Server")
    print("Server will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
