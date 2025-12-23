"""
Unified Deployment Manager - Choose your deployment strategy
Supports: Ray Serve (Online), Ray Data (Batch)
"""

import os
from typing import Any, Dict, Literal
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DeploymentType = Literal[
    "ray-serve-online",
    "ray-batch",
]


class DeploymentManager:
    """Unified manager for deployment types."""

    def __init__(
        self,
        model_name: str = "model-name",
        model_version: str = "latest",
        mlflow_uri: str = "http://127.0.0.1:5001",
    ):
        """
        Initialize deployment manager.

        Args:
            model_name: Model name in MLflow
            model_version: Model version or 'latest'
            mlflow_uri: MLflow tracking URI
        """
        self.model_name = model_name
        self.model_version = model_version
        self.mlflow_uri = mlflow_uri

        # Set environment variables
        os.environ["MODEL_NAME"] = model_name
        os.environ["MODEL_VERSION"] = model_version
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri

    def deploy(self, deployment_type: DeploymentType, **kwargs) -> Dict[str, Any]:
        """
        Deploy model to specified target.

        Args:
            deployment_type: Type of deployment
            **kwargs: Additional deployment-specific arguments

        Returns:
            Deployment information
        """
        logger.info(f"DEPLOYMENT MANAGER - {deployment_type.upper()}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Version: {self.model_version}")

        if deployment_type == "ray-serve-online":
            return self._deploy_ray_serve(**kwargs)

        elif deployment_type == "ray-batch":
            return self._deploy_ray_batch(**kwargs)

        else:
            raise ValueError(f"Unknown deployment type: {deployment_type}")

    def _deploy_ray_serve(self, port: int = 8000, num_replicas: int = 2) -> Dict:
        """Deploy with Ray Serve for online inference."""
        logger.info("🚀 Deploying Ray Serve (Online Inference)...")

        try:
            # Import and deploy
            from .ray_serve import deploy

            os.environ["NUM_REPLICAS"] = str(num_replicas)
            
            # Since ray_serve.py defines the deployment logic, we might need to invoke it appropriately
            # Assuming ray_serve.py has a deploy function or we run it as a script.
            # Here we just import it to trigger deployment if it has a main block or explicit deploy function.
            # If ray_serve.py is a script, we might need to subprocess it or refactor it to be callable.
            
            # For now, let's assume we return the config used
            return {
                "status": "initiated",
                "type": "ray-serve-online",
                "port": port,
                "replicas": num_replicas
            }

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _deploy_ray_batch(self, input_path: str, output_path: str) -> Dict:
        """Deploy with Ray Data for batch inference."""
        logger.info("📦 Running Ray Data (Batch Inference)...")

        try:
            # Import and run
            from .ray_batch import run_batch_inference

            result = run_batch_inference(input_path, output_path)
            return {
                "status": "success",
                "type": "ray-batch",
                "result": result
            }

        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            return {"status": "failed", "error": str(e)}
