"""
Ray Serve deployment for online inference.
Supports: On-premise servers, Cloud VMs (AWS EC2, GCP Compute, Azure VMs)

Features:
- Auto-scaling based on load
- High availability with replicas
- GPU support
- A/B testing ready
- Prometheus metrics
"""

import os
from typing import Any, Dict, List

import mlflow
import numpy as np
import pandas as pd
import ray
from pydantic import BaseModel
from ray import serve

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = os.getenv("MODEL_NAME", "iris-classifier")
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
NUM_REPLICAS = int(os.getenv("NUM_REPLICAS", "2"))
MAX_CONCURRENT_QUERIES = int(os.getenv("MAX_CONCURRENT_QUERIES", "100"))


# =============================================================================
# Pydantic Models
# =============================================================================


class PredictionRequest(BaseModel):
    """Request schema for predictions."""

    features: List[List[float]]
    feature_names: List[str] = None


class PredictionResponse(BaseModel):
    """Response schema for predictions."""

    predictions: List[int]
    probabilities: List[List[float]] = None
    model_name: str
    model_version: str


# =============================================================================
# Ray Serve Deployment
# =============================================================================


@serve.deployment(
    name="model-inference",
    num_replicas=NUM_REPLICAS,
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 0,  # Set to 1 if using GPU
    },
    max_concurrent_queries=MAX_CONCURRENT_QUERIES,
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_num_ongoing_requests_per_replica": 5,
    },
    health_check_period_s=10,
    health_check_timeout_s=30,
)
class MLModelDeployment:
    """Ray Serve deployment for ML model inference."""

    def __init__(self):
        """Initialize model on deployment startup."""
        self.model_name = MODEL_NAME
        self.model_version = MODEL_VERSION
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load model from MLflow."""
        print(f"Loading model: {self.model_name} (version: {self.model_version})")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        try:
            if self.model_version == "latest":
                model_uri = f"models:/{self.model_name}@champion"
            else:
                model_uri = f"models:/{self.model_name}/{self.model_version}"

            self.model = mlflow.pyfunc.load_model(model_uri)
            print(f"✓ Model loaded successfully from {model_uri}")

        except Exception as e:
            print(f"⚠️ Failed to load model from MLflow: {e}")
            # Fallback to local model
            try:
                local_path = f"models/{self.model_name}"
                self.model = mlflow.pyfunc.load_model(local_path)
                print(f"✓ Model loaded from local path: {local_path}")
            except Exception as e2:
                print(f"❌ Failed to load model: {e2}")
                raise

    async def __call__(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle inference request.

        Args:
            request: Dictionary with 'features' and optional 'feature_names'

        Returns:
            Dictionary with predictions and metadata
        """
        try:
            # Parse request
            features = request.get("features", [])
            feature_names = request.get("feature_names", None)

            if not features:
                return {"error": "No features provided", "status": "error"}

            # Convert to DataFrame
            if feature_names:
                df = pd.DataFrame(features, columns=feature_names)
            else:
                df = pd.DataFrame(features)

            # Make predictions
            predictions = self.model.predict(df)

            # Get probabilities if available
            probabilities = None
            try:
                if hasattr(self.model._model_impl, "predict_proba"):
                    probabilities = self.model._model_impl.predict_proba(df).tolist()
            except:
                pass

            # Build response
            response = {
                "predictions": (
                    predictions.tolist() if hasattr(predictions, "tolist") else predictions
                ),
                "probabilities": probabilities,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "status": "success",
            }

            return response

        except Exception as e:
            return {"error": str(e), "status": "error"}

    def health_check(self) -> bool:
        """Health check endpoint."""
        return self.model is not None


# =============================================================================
# Application Setup
# =============================================================================


def build_app():
    """Build Ray Serve application."""
    return MLModelDeployment.bind()


# =============================================================================
# Deployment Functions
# =============================================================================


def deploy(blocking: bool = True):
    """
    Deploy model to Ray Serve.

    Args:
        blocking: If True, blocks until deployment is ready
    """
    print("\n" + "=" * 70)
    print("RAY SERVE DEPLOYMENT - ONLINE INFERENCE")
    print("=" * 70)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(address=os.getenv("RAY_ADDRESS", "auto"), ignore_reinit_error=True)

    # Start Ray Serve
    serve.start(
        detached=True,
        http_options={
            "host": "0.0.0.0",
            "port": 8000,
        },
    )

    # Deploy the model
    app = build_app()
    handle = serve.run(app, name="model-inference", route_prefix="/predict")

    print(f"\n✓ Deployment successful!")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Version: {MODEL_VERSION}")
    print(f"  Replicas: {NUM_REPLICAS}")
    print(f"  Endpoint: http://localhost:8000/predict")
    print("=" * 70 + "\n")

    return handle


def shutdown():
    """Shutdown Ray Serve and Ray."""
    print("Shutting down Ray Serve...")
    serve.shutdown()
    ray.shutdown()
    print("✓ Shutdown complete")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deploy model with Ray Serve")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--model-version", type=str, default=MODEL_VERSION)
    parser.add_argument("--num-replicas", type=int, default=NUM_REPLICAS)
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    # Update environment
    os.environ["MODEL_NAME"] = args.model_name
    os.environ["MODEL_VERSION"] = args.model_version
    os.environ["NUM_REPLICAS"] = str(args.num_replicas)

    # Deploy
    try:
        handle = deploy(blocking=True)

        # Keep running
        print("\nPress Ctrl+C to stop...\n")
        import signal

        signal.pause()

    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    finally:
        shutdown()


# =============================================================================
# Example Usage
# =============================================================================
"""
# Start Ray Serve deployment
python src/deployment/ray_serve_online.py \
    --model-name iris-classifier \
    --model-version latest \
    --num-replicas 2

# Test with curl
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]],
    "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
  }'

# Test with Python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "features": [[5.1, 3.5, 1.4, 0.2]],
        "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    }
)
print(response.json())

# Monitor with Ray Dashboard
# http://localhost:8265
"""
