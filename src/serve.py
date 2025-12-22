"""
FastAPI inference service for ML models.
Serves predictions via REST API with MLflow model loading.
"""

import os
from datetime import datetime
from typing import Any, Dict, List

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = os.getenv("MODEL_NAME", "iris-classifier")
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# =============================================================================
# Pydantic Models
# =============================================================================


class PredictionRequest(BaseModel):
    """Request model for predictions."""

    features: List[List[float]] = Field(
        ..., description="2D array of feature values", example=[[5.1, 3.5, 1.4, 0.2]]
    )
    feature_names: List[str] = Field(
        default=None,
        description="Optional feature names",
        example=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    )


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    predictions: List[int]
    probabilities: List[List[float]] = None
    model_name: str
    model_version: str
    prediction_time: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    timestamp: str


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="ML Model Inference API",
    description="REST API for ML model predictions using MLflow",
    version="1.0.0",
)

# Global model variable
model = None


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model

    try:
        print(f"Loading model: {MODEL_NAME} (version: {MODEL_VERSION})")

        if MODEL_VERSION == "latest":
            # Load latest version with champion alias
            model_uri = f"models:/{MODEL_NAME}@champion"
        else:
            # Load specific version
            model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

        model = mlflow.pyfunc.load_model(model_uri)
        print(f"✓ Model loaded successfully from {model_uri}")

    except Exception as e:
        print(f"⚠️ Failed to load model from MLflow: {e}")
        print("Attempting to load from local file...")

        try:
            # Fallback: load from local file
            local_path = f"models/{MODEL_NAME}"
            model = mlflow.pyfunc.load_model(local_path)
            print(f"✓ Model loaded from local path: {local_path}")
        except Exception as e2:
            print(f"❌ Failed to load model: {e2}")
            model = None


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "ML Model Inference API",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "endpoints": {"health": "/health", "predict": "/predict", "docs": "/docs"},
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make predictions on input data.

    Args:
        request: Prediction request with features

    Returns:
        Predictions and optional probabilities
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")

    try:
        start_time = datetime.now()

        # Convert to DataFrame
        if request.feature_names:
            df = pd.DataFrame(request.features, columns=request.feature_names)
        else:
            df = pd.DataFrame(request.features)

        # Make predictions
        predictions = model.predict(df)

        # Get probabilities if available
        probabilities = None
        try:
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(df).tolist()
        except:
            pass

        prediction_time = (datetime.now() - start_time).total_seconds()

        return PredictionResponse(
            predictions=predictions.tolist() if hasattr(predictions, "tolist") else predictions,
            probabilities=probabilities,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            prediction_time=f"{prediction_time:.4f}s",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=PredictionResponse, tags=["Prediction"])
async def predict_batch(request: PredictionRequest):
    """
    Batch prediction endpoint (alias for /predict).

    Args:
        request: Prediction request with features

    Returns:
        Predictions and optional probabilities
    """
    return await predict(request)


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Try to get model metadata
        info = {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
            "model_loaded": True,
        }

        # Add model-specific info if available
        if hasattr(model, "metadata"):
            info["metadata"] = model.metadata.to_dict()

        return info

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


# =============================================================================
# Example Usage
# =============================================================================
"""
# Start the server:
uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload

# Test with curl:

# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[5.1, 3.5, 1.4, 0.2]],
    "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
  }'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [5.1, 3.5, 1.4, 0.2],
      [6.2, 3.4, 5.4, 2.3]
    ]
  }'

# Model info
curl http://localhost:8000/model/info
"""
