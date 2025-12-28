"""
Ray Serve deployment for online inference.
Fully integrated with MLflow - loads champion model automatically.
"""
import os
import logging
from typing import List, Dict

from ray import serve
import mlflow
from fastapi import FastAPI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment
MODEL_NAME = os.getenv("MODEL_NAME", "model-name")
MODEL_VERSION = os.getenv("MODEL_VERSION", "champion")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
SERVE_MIN_REPLICAS = int(os.getenv("SERVE_MIN_REPLICAS", "1"))
SERVE_MAX_REPLICAS = int(os.getenv("SERVE_MAX_REPLICAS", "4"))
SERVE_TARGET_CONCURRENCY = int(os.getenv("SERVE_TARGET_CONCURRENCY", "8"))
SERVE_NUM_CPUS_PER_REPLICA = float(os.getenv("SERVE_NUM_CPUS_PER_REPLICA", "1"))

app = FastAPI()


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
    """Ray Serve deployment that loads models from MLflow."""
    
    def __init__(self):
        """Initialize and load model from MLflow."""
        self.model = None
        self.model_uri = None
        self._load_model()
    
    def _load_model(self):
        """Load model from MLflow using champion alias."""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        try:
            # Always try to load with alias first (champion, staging, production)
            if MODEL_VERSION.lower() in ("champion", "staging", "production"):
                self.model_uri = f"models:/{MODEL_NAME}@{MODEL_VERSION.lower()}"
                logger.info(f"Loading model with alias: {self.model_uri}")
            elif MODEL_VERSION.lower() == "latest":
                self.model_uri = f"models:/{MODEL_NAME}/latest"
                logger.info(f"Loading latest model version: {self.model_uri}")
            else:
                # Load specific version number
                self.model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
                logger.info(f"Loading model version: {self.model_uri}")
            
            self.model = mlflow.pyfunc.load_model(self.model_uri)
            logger.info(f"✅ Model loaded successfully: {self.model_uri}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            logger.info("💡 Ensure model is registered with alias '@champion' in MLflow")
            logger.info("   Or specify a version number via MODEL_VERSION env var")
            self.model = None
    
    @app.post("/predict")
    async def predict(self, features: List[List[float]]) -> Dict:
        """
        Make predictions on input features.
        
        Args:
            features: List of feature vectors
            
        Returns:
            Dictionary with predictions and metadata
        """
        if not self.model:
            return {
                "error": "Model not loaded",
                "message": "Model is not available. Check logs for details."
            }
        
        try:
            predictions = self.model.predict(features)
            return {
                "predictions": predictions.tolist(),
                "model_uri": self.model_uri,
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": str(e)}
    
    @app.get("/health")
    def health(self) -> Dict:
        """Health check endpoint."""
        return {
            "status": "healthy" if self.model else "unhealthy",
            "model_loaded": self.model is not None,
            "model_uri": self.model_uri,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION
        }
    
    @app.get("/info")
    def info(self) -> Dict:
        """Get deployment information."""
        return {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "model_uri": self.model_uri,
            "mlflow_uri": MLFLOW_TRACKING_URI,
            "replicas": {
                "min": SERVE_MIN_REPLICAS,
                "max": SERVE_MAX_REPLICAS
            }
        }


# Entrypoint for Ray Serve
entrypoint = MLFlowDeployment.bind()


# =============================================================================
# Usage Examples
# =============================================================================
"""
# Start Ray Serve
serve run src.deployment.ray_serve:entrypoint

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[1.0, 2.0, 3.0]]}'

# Health check
curl http://localhost:8000/health

# Get info
curl http://localhost:8000/info
"""
