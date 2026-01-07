"""
Ray Serve deployment with robust MLflow integration.
Handles network issues, retries, and proper error handling for production.
"""

import logging
import os
import time
from typing import Dict, List

import mlflow
from fastapi import FastAPI, HTTPException
from ray import serve

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration from environment
MODEL_NAME = os.getenv("MODEL_NAME", "model-name")
MODEL_VERSION = os.getenv("MODEL_VERSION", "champion")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
SERVE_MIN_REPLICAS = int(os.getenv("SERVE_MIN_REPLICAS", "1"))
SERVE_MAX_REPLICAS = int(os.getenv("SERVE_MAX_REPLICAS", "4"))
SERVE_TARGET_CONCURRENCY = int(os.getenv("SERVE_TARGET_CONCURRENCY", "8"))
SERVE_NUM_CPUS_PER_REPLICA = float(os.getenv("SERVE_NUM_CPUS_PER_REPLICA", "1"))

app = FastAPI(
    title="MLflow Model Serving",
    description="Ray Serve deployment with MLflow model registry integration",
    version="1.0.0",
)


@serve.deployment(
    ray_actor_options={"num_cpus": SERVE_NUM_CPUS_PER_REPLICA},
    autoscaling_config={
        "min_replicas": SERVE_MIN_REPLICAS,
        "max_replicas": SERVE_MAX_REPLICAS,
        "target_num_ongoing_requests_per_replica": SERVE_TARGET_CONCURRENCY,
    },
)
@serve.ingress(app)
class MLFlowDeployment:
    """Ray Serve deployment with robust model loading and error handling."""

    def __init__(self):
        """Initialize with retry logic for model loading."""
        self.model = None
        self.model_uri = None
        self.model_metadata = {}

        logger.info("=" * 70)
        logger.info("INITIALIZING MLFLOW DEPLOYMENT")
        logger.info("=" * 70)
        logger.info(f"Model Name: {MODEL_NAME}")
        logger.info(f"Model Version: {MODEL_VERSION}")
        logger.info(f"MLflow URI: {MLFLOW_TRACKING_URI}")
        logger.info("=" * 70)

        self._load_model_with_retry(max_retries=10, initial_delay=5)

    def _load_model_with_retry(self, max_retries: int = 10, initial_delay: int = 5):
        """Load model with exponential backoff retry logic."""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        for attempt in range(max_retries):
            try:
                logger.info(f"\nAttempt {attempt + 1}/{max_retries}: Loading model...")

                # Construct model URI based on version type
                if MODEL_VERSION.lower() in ("champion", "staging", "production", "archived"):
                    # Using alias
                    self.model_uri = f"models:/{MODEL_NAME}@{MODEL_VERSION.lower()}"
                    logger.info(f"Using alias: {MODEL_VERSION}")
                elif MODEL_VERSION.lower() == "latest":
                    # Using latest version
                    self.model_uri = f"models:/{MODEL_NAME}/latest"
                    logger.info("Using latest version")
                elif MODEL_VERSION.isdigit():
                    # Using specific version number
                    self.model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
                    logger.info(f"Using version number: {MODEL_VERSION}")
                else:
                    raise ValueError(f"Invalid MODEL_VERSION: {MODEL_VERSION}")

                logger.info(f"Model URI: {self.model_uri}")

                # Load model from MLflow
                self.model = mlflow.pyfunc.load_model(self.model_uri)

                # Get model metadata
                try:
                    client = mlflow.tracking.MlflowClient()

                    if "@" in self.model_uri:
                        # Using alias - get version info
                        model_name_part = self.model_uri.split("@")[0].replace("models:/", "")
                        alias = MODEL_VERSION.lower()
                        version_info = client.get_model_version_by_alias(model_name_part, alias)

                        self.model_metadata = {
                            "model_name": model_name_part,
                            "version": version_info.version,
                            "alias": alias,
                            "stage": version_info.current_stage,
                            "run_id": version_info.run_id,
                        }
                    elif "/latest" in self.model_uri:
                        # Latest version
                        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
                        if versions:
                            latest = max(versions, key=lambda v: int(v.version))
                            self.model_metadata = {
                                "model_name": MODEL_NAME,
                                "version": latest.version,
                                "stage": latest.current_stage,
                                "run_id": latest.run_id,
                            }
                    else:
                        # Specific version
                        self.model_metadata = {
                            "model_name": MODEL_NAME,
                            "version": MODEL_VERSION,
                        }

                except Exception as meta_error:
                    logger.warning(f"Could not fetch model metadata: {meta_error}")

                # Success!
                logger.info("=" * 70)
                logger.info("✅ MODEL LOADED SUCCESSFULLY")
                logger.info("=" * 70)
                logger.info(f"Model URI: {self.model_uri}")
                if self.model_metadata:
                    for key, value in self.model_metadata.items():
                        logger.info(f"{key.capitalize()}: {value}")
                logger.info("=" * 70)

                return  # Exit retry loop

            except Exception as e:
                logger.error(f"❌ Attempt {attempt + 1} failed: {str(e)}")

                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = initial_delay * (2**attempt)
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # All retries exhausted
                    logger.error("=" * 70)
                    logger.error("❌ ALL RETRY ATTEMPTS EXHAUSTED")
                    logger.error("=" * 70)
                    logger.error(f"MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")
                    logger.error(f"MODEL_NAME: {MODEL_NAME}")
                    logger.error(f"MODEL_VERSION: {MODEL_VERSION}")
                    logger.error("=" * 70)
                    logger.error("Deployment will continue but predictions will fail.")
                    logger.error("Check MLflow connectivity and model registration.")
                    logger.error("=" * 70)
                    # Don't raise - allow deployment to start in degraded state

    @app.post("/predict")
    async def predict(self, features: List[List[float]]) -> Dict:
        """
        Make predictions on input features.

        Request body:
        {
            "features": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        }

        Response:
        {
            "predictions": [0, 1],
            "model_uri": "models:/my-model@champion",
            "model_info": {...}
        }
        """
        if not self.model:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Service is in degraded state. Check logs.",
            )

        try:
            predictions = self.model.predict(features)

            return {
                "predictions": predictions.tolist(),
                "model_uri": self.model_uri,
                "model_metadata": self.model_metadata,
            }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    @app.get("/health")
    def health(self) -> Dict:
        """
        Health check endpoint.

        Returns:
        {
            "status": "healthy" | "unhealthy",
            "model_loaded": true | false,
            "model_uri": "models:/my-model@champion",
            ...
        }
        """
        is_healthy = self.model is not None

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "model_loaded": is_healthy,
            "model_uri": self.model_uri,
            "model_metadata": self.model_metadata,
            "mlflow_uri": MLFLOW_TRACKING_URI,
        }

    @app.get("/info")
    def info(self) -> Dict:
        """
        Deployment information endpoint.

        Returns detailed configuration and model information.
        """
        return {
            "service": "MLflow Model Serving",
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "model_uri": self.model_uri,
            "model_metadata": self.model_metadata,
            "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
            "autoscaling": {
                "min_replicas": SERVE_MIN_REPLICAS,
                "max_replicas": SERVE_MAX_REPLICAS,
                "target_concurrency": SERVE_TARGET_CONCURRENCY,
            },
            "resources": {
                "cpus_per_replica": SERVE_NUM_CPUS_PER_REPLICA,
            },
        }

    @app.get("/")
    def root(self) -> Dict:
        """Root endpoint with API documentation links."""
        return {
            "message": "MLflow Model Serving API",
            "endpoints": {
                "predict": "POST /predict",
                "health": "GET /health",
                "info": "GET /info",
                "docs": "GET /docs",
            },
        }


# Entrypoint for Ray Serve
entrypoint = MLFlowDeployment.bind()


# =============================================================================
# Usage Examples
# =============================================================================
"""
# Local development:
serve run src.deployment.ray_serve:entrypoint

# With custom config:
MODEL_NAME=my-model MODEL_VERSION=champion serve run src.deployment.ray_serve:entrypoint

# Docker:
docker run -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
           -e MODEL_NAME=my-model \
           -e MODEL_VERSION=champion \
           -p 8000:8000 \
           my-image:latest

# Test prediction:
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[1.0, 2.0, 3.0]]}'

# Health check:
curl http://localhost:8000/health

# Info:
curl http://localhost:8000/info
"""
