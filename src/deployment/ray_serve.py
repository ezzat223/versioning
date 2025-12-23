"""
Ray Serve deployment for online inference.
Fully integrated with MLflow and CI/CD pipeline.

Features:
- Auto-loads champion model from MLflow
- Auto-scaling based on load
- Health checks and metrics
- Rolling updates support
- A/B testing ready
"""
import os
from typing import Dict, List, Any, Optional

import ray
from ray import serve
from ray.serve.handle import DeploymentHandle
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from pydantic import BaseModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = os.getenv("MODEL_NAME", "iris-classifier")
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")  # or "champion"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
NUM_REPLICAS = int(os.getenv("NUM_REPLICAS", "2"))
MAX_CONCURRENT_QUERIES = int(os.getenv("MAX_CONCURRENT_QUERIES", "100"))


# =============================================================================
# Pydantic Models for API
# =============================================================================

class PredictionRequest(BaseModel):
    """Request schema for predictions."""
    features: List[List[float]]
    feature_names: Optional[List[str]] = None


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    predictions: List[int]
    probabilities: Optional[List[List[float]]] = None
    model_name: str
    model_version: str
    model_alias: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    model_alias: str
    num_replicas: int


# =============================================================================
# Model Loading Utilities
# =============================================================================

def get_champion_model_info() -> Dict[str, str]:
    """
    Get champion model information from MLflow.
    
    Returns:
        Dictionary with model_name, model_version, run_id
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    try:
        # Search for champion model
        experiment = mlflow.get_experiment_by_name(f"{MODEL_NAME.replace('-', '_')}-ci")
        if not experiment:
            # Try without -ci suffix
            experiment = mlflow.get_experiment_by_name(MODEL_NAME.replace("-", "_"))
        
        if not experiment:
            raise ValueError(f"No experiment found for {MODEL_NAME}")
        
        # Get champion run
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.model_alias = 'champion'",
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs:
            run = runs[0]
            return {
                "run_id": run.info.run_id,
                "model_name": MODEL_NAME,
                "model_alias": "champion",
                "accuracy": run.data.metrics.get("test_accuracy", 0.0)
            }
        else:
            # No champion yet, get latest run
            logger.warning("No champion model found, using latest run")
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if runs:
                run = runs[0]
                return {
                    "run_id": run.info.run_id,
                    "model_name": MODEL_NAME,
                    "model_alias": "latest",
                    "accuracy": run.data.metrics.get("test_accuracy", 0.0)
                }
            else:
                raise ValueError("No models found in MLflow")
    
    except Exception as e:
        logger.error(f"Error getting champion model: {e}")
        raise


# =============================================================================
# Ray Serve Deployment
# =============================================================================

@serve.deployment(
    name="ml-model",
    num_replicas=NUM_REPLICAS,
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 0,  # Set to 1 for GPU inference
    },
    max_concurrent_queries=MAX_CONCURRENT_QUERIES,
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_num_ongoing_requests_per_replica": 5,
        "upscale_delay_s": 30,
        "downscale_delay_s": 600,
    },
    health_check_period_s=10,
    health_check_timeout_s=30,
)
class MLModelDeployment:
    """Ray Serve deployment for ML model inference."""
    
    def __init__(self):
        """Initialize model on deployment startup."""
        logger.info("Initializing ML Model Deployment...")
        
        self.model_name = MODEL_NAME
        self.model_version = MODEL_VERSION
        self.mlflow_uri = MLFLOW_TRACKING_URI
        self.model = None
        self.model_info = {}
        
        # Load model
        self._load_model()
        
        logger.info(f"✓ Deployment initialized: {self.model_info}")
    
    def _load_model(self):
        """Load champion model from MLflow."""
        logger.info(f"Loading model: {self.model_name}")
        
        mlflow.set_tracking_uri(self.mlflow_uri)
        
        try:
            # Get champion model info
            self.model_info = get_champion_model_info()
            
            # Construct model URI
            if self.model_version == "latest" or self.model_version == "champion":
                # Try registered model first
                try:
                    model_uri = f"models:/{self.model_name}@champion"
                    self.model = mlflow.pyfunc.load_model(model_uri)
                    logger.info(f"✓ Loaded from Model Registry: {model_uri}")
                except:
                    # Fallback to run URI
                    model_uri = f"runs:/{self.model_info['run_id']}/model"
                    self.model = mlflow.pyfunc.load_model(model_uri)
                    logger.info(f"✓ Loaded from run: {model_uri}")
            else:
                # Load specific version
                model_uri = f"models:/{self.model_name}/{self.model_version}"
                self.model = mlflow.pyfunc.load_model(model_uri)
                logger.info(f"✓ Loaded specific version: {model_uri}")
            
            logger.info(f"Model accuracy: {self.model_info.get('accuracy', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Try local fallback
            try:
                local_path = f"models/{self.model_name}"
                self.model = mlflow.pyfunc.load_model(local_path)
                logger.warning(f"✓ Loaded from local: {local_path}")
                self.model_info = {
                    "model_name": self.model_name,
                    "model_alias": "local",
                    "run_id": "local"
                }
            except Exception as e2:
                logger.error(f"Local fallback failed: {e2}")
                raise RuntimeError(f"Could not load model: {e}")
    
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
                return {
                    "error": "No features provided",
                    "status": "error"
                }
            
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
                if hasattr(self.model._model_impl, 'predict_proba'):
                    probabilities = self.model._model_impl.predict_proba(df).tolist()
            except:
                pass
            
            # Build response
            response = {
                "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
                "probabilities": probabilities,
                "model_name": self.model_info.get("model_name", self.model_name),
                "model_version": self.model_info.get("run_id", "unknown"),
                "model_alias": self.model_info.get("model_alias", "unknown"),
                "status": "success"
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    def health_check(self) -> bool:
        """Health check for Ray Serve."""
        return self.model is not None


# =============================================================================
# FastAPI-style endpoints using Ray Serve
# =============================================================================

@serve.deployment(route_prefix="/")
@serve.ingress(serve.http.FastAPI)
class APIGateway:
    """API Gateway with FastAPI for Ray Serve."""
    
    def __init__(self, model_handle: DeploymentHandle):
        """Initialize with model deployment handle."""
        self.model = model_handle
    
    @serve.http.get("/")
    async def root(self):
        """Root endpoint."""
        return {
            "service": "ML Model Inference API",
            "model": MODEL_NAME,
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "predict": "/predict",
                "model_info": "/model/info"
            }
        }
    
    @serve.http.get("/health")
    async def health(self):
        """Health check endpoint."""
        try:
            # Call model's health check
            is_healthy = await self.model.health_check.remote()
            
            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "model_loaded": is_healthy,
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "num_replicas": NUM_REPLICAS
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    @serve.http.post("/predict")
    async def predict(self, request: PredictionRequest):
        """Prediction endpoint."""
        try:
            # Call model deployment
            result = await self.model.remote(request.dict())
            return result
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }
    
    @serve.http.get("/model/info")
    async def model_info(self):
        """Get model information."""
        try:
            model_info = get_champion_model_info()
            return {
                "model_name": MODEL_NAME,
                "model_info": model_info,
                "mlflow_uri": MLFLOW_TRACKING_URI,
                "num_replicas": NUM_REPLICAS
            }
        except Exception as e:
            return {
                "error": str(e)
            }


# =============================================================================
# Application Builder
# =============================================================================

def build_app():
    """Build Ray Serve application."""
    model_deployment = MLModelDeployment.bind()
    api_gateway = APIGateway.bind(model_deployment)
    return api_gateway


# =============================================================================
# Deployment Functions
# =============================================================================

def deploy(blocking: bool = True) -> str:
    """
    Deploy model to Ray Serve.
    
    Args:
        blocking: If True, blocks until deployment is ready
        
    Returns:
        Deployment status message
    """
    logger.info("\n" + "="*70)
    logger.info("RAY SERVE DEPLOYMENT")
    logger.info("="*70)
    
    # Initialize Ray if not already
    if not ray.is_initialized():
        ray_address = os.getenv("RAY_ADDRESS", "auto")
        logger.info(f"Initializing Ray (address: {ray_address})...")
        ray.init(
            address=ray_address,
            ignore_reinit_error=True,
            logging_level=logging.INFO
        )
    
    # Start Ray Serve
    logger.info("Starting Ray Serve...")
    serve.start(
        detached=True,
        http_options={
            "host": "0.0.0.0",
            "port": 8000,
        }
    )
    
    # Build and deploy application
    logger.info("Deploying application...")
    app = build_app()
    serve.run(
        app,
        name="ml-model-api",
        route_prefix="/"
    )
    
    logger.info(f"\n✓ Deployment successful!")
    logger.info(f"  Model: {MODEL_NAME}")
    logger.info(f"  Version: {MODEL_VERSION}")
    logger.info(f"  Replicas: {NUM_REPLICAS}")
    logger.info(f"  Endpoint: http://localhost:8000")
    logger.info(f"  Dashboard: http://localhost:8265")
    logger.info("="*70 + "\n")
    
    return "Deployment successful"


def shutdown():
    """Shutdown Ray Serve and Ray."""
    logger.info("Shutting down Ray Serve...")
    serve.shutdown()
    ray.shutdown()
    logger.info("✓ Shutdown complete")


def update_model():
    """
    Update deployed model to latest champion.
    Performs rolling update with zero downtime.
    """
    logger.info("Updating model to latest champion...")
    
    # Ray Serve automatically does rolling update
    # Just redeploy with new model
    app = build_app()
    serve.run(
        app,
        name="ml-model-api",
        route_prefix="/"
    )
    
    logger.info("✓ Model updated with zero downtime")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy ML model with Ray Serve")
    parser.add_argument("command", choices=["deploy", "update", "shutdown"], 
                       help="Command to execute")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--model-version", type=str, default=MODEL_VERSION)
    parser.add_argument("--num-replicas", type=int, default=NUM_REPLICAS)
    
    args = parser.parse_args()
    
    # Update environment
    os.environ["MODEL_NAME"] = args.model_name
    os.environ["MODEL_VERSION"] = args.model_version
    os.environ["NUM_REPLICAS"] = str(args.num_replicas)
    
    # Execute command
    if args.command == "deploy":
        try:
            deploy()
            logger.info("\nPress Ctrl+C to stop...\n")
            import signal
            signal.pause()
        except KeyboardInterrupt:
            logger.info("\nReceived interrupt signal")
        finally:
            shutdown()
    
    elif args.command == "update":
        update_model()
    
    elif args.command == "shutdown":
        shutdown()


# =============================================================================
# Example Usage
# =============================================================================
"""
# Deploy
python src/deployment/ray_serve.py deploy \
    --model-name iris-classifier \
    --num-replicas 2

# Update to latest champion (zero downtime)
python src/deployment/ray_serve.py update

# Test
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[5.1, 3.5, 1.4, 0.2]],
    "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
  }'

# Monitor
open http://localhost:8265  # Ray Dashboard

# Shutdown
python src/deployment/ray_serve.py shutdown
"""
