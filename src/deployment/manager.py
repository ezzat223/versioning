"""
Unified Deployment Manager - Choose your deployment strategy
Supports: Ray Serve, Ray Batch, BentoML, Multi-cloud
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Literal

DeploymentType = Literal[
    "ray-serve-online",
    "ray-batch",
    "bentoml-local",
    "bentoml-aws-lambda",
    "bentoml-gcp-cloudrun",
    "bentoml-azure-functions",
    "docker-compose",
]


class DeploymentManager:
    """Unified manager for all deployment types."""

    def __init__(
        self,
        model_name: str = "iris-classifier",
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
        print(f"\n{'='*70}")
        print(f"DEPLOYMENT MANAGER - {deployment_type.upper()}")
        print(f"{'='*70}")
        print(f"Model: {self.model_name}")
        print(f"Version: {self.model_version}")
        print(f"{'='*70}\n")

        if deployment_type == "ray-serve-online":
            return self._deploy_ray_serve(**kwargs)

        elif deployment_type == "ray-batch":
            return self._deploy_ray_batch(**kwargs)

        elif deployment_type == "bentoml-local":
            return self._deploy_bentoml_local(**kwargs)

        elif deployment_type == "bentoml-aws-lambda":
            return self._deploy_bentoml_aws(**kwargs)

        elif deployment_type == "bentoml-gcp-cloudrun":
            return self._deploy_bentoml_gcp(**kwargs)

        elif deployment_type == "bentoml-azure-functions":
            return self._deploy_bentoml_azure(**kwargs)

        elif deployment_type == "docker-compose":
            return self._deploy_docker_compose(**kwargs)

        else:
            raise ValueError(f"Unknown deployment type: {deployment_type}")

    def _deploy_ray_serve(self, port: int = 8000, num_replicas: int = 2) -> Dict:
        """Deploy with Ray Serve for online inference."""
        print("🚀 Deploying Ray Serve (Online Inference)...")

        try:
            # Import and deploy
            from src.deployment.ray_serve_online import deploy

            os.environ["NUM_REPLICAS"] = str(num_replicas)

            handle = deploy(blocking=False)

            return {
                "status": "success",
                "deployment_type": "ray-serve-online",
                "endpoint": f"http://localhost:{port}/predict",
                "dashboard": f"http://localhost:8265",
                "num_replicas": num_replicas,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _deploy_ray_batch(self, **kwargs) -> Dict:
        """Setup Ray for batch inference."""
        print("📊 Setting up Ray Batch Inference...")

        try:
            from src.deployment.ray_batch_inference import RayBatchInference

            pipeline = RayBatchInference(
                model_name=self.model_name,
                model_version=self.model_version,
                tracking_uri=self.mlflow_uri,
            )

            return {
                "status": "success",
                "deployment_type": "ray-batch",
                "message": "Ray batch pipeline initialized",
                "usage": "Use pipeline.predict_dataframe() or pipeline.predict_csv()",
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _deploy_bentoml_local(self, port: int = 3000) -> Dict:
        """Deploy BentoML service locally."""
        print("🍱 Deploying BentoML Service Locally...")

        try:
            # Load model to BentoML first
            subprocess.run(
                [sys.executable, "src/deployment/bentoml_service.py", "load"], check=True
            )

            # Start server in background
            print(f"\nStarting BentoML server on port {port}...")
            process = subprocess.Popen(
                ["bentoml", "serve", "src.deployment.bentoml_service:svc", "--port", str(port)]
            )

            return {
                "status": "success",
                "deployment_type": "bentoml-local",
                "endpoint": f"http://localhost:{port}/predict",
                "docs": f"http://localhost:{port}/docs",
                "process_id": process.pid,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _deploy_bentoml_aws(self, **kwargs) -> Dict:
        """Deploy to AWS Lambda using BentoML + bentoctl."""
        print("☁️ Deploying to AWS Lambda...")

        try:
            # Build Bento
            print("1. Building Bento...")
            subprocess.run(
                [sys.executable, "src/deployment/bentoml_service.py", "build"], check=True
            )

            # Get latest bento tag
            result = subprocess.run(
                ["bentoml", "list", "--output", "json"], capture_output=True, text=True, check=True
            )

            import json

            bentos = json.loads(result.stdout)
            latest_tag = bentos[0]["tag"] if bentos else None

            if not latest_tag:
                raise ValueError("No Bento found")

            print(f"2. Deploying Bento: {latest_tag}")

            # Deploy with bentoctl
            subprocess.run(
                [
                    "bentoctl",
                    "build",
                    "-b",
                    latest_tag,
                    "-f",
                    "deployment/deployment_configs.yaml:aws_lambda",
                ],
                check=True,
            )

            subprocess.run(["bentoctl", "deploy", "-b", latest_tag], check=True)

            return {
                "status": "success",
                "deployment_type": "bentoml-aws-lambda",
                "bento_tag": latest_tag,
                "message": "Check AWS Lambda console for endpoint URL",
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _deploy_bentoml_gcp(self, **kwargs) -> Dict:
        """Deploy to GCP Cloud Run using BentoML."""
        print("☁️ Deploying to GCP Cloud Run...")

        try:
            # Similar to AWS Lambda deployment
            # Build and deploy using bentoctl

            subprocess.run(
                [sys.executable, "src/deployment/bentoml_service.py", "build"], check=True
            )

            # Get latest bento
            result = subprocess.run(
                ["bentoml", "list", "--output", "json"], capture_output=True, text=True, check=True
            )

            import json

            bentos = json.loads(result.stdout)
            latest_tag = bentos[0]["tag"] if bentos else None

            subprocess.run(
                [
                    "bentoctl",
                    "build",
                    "-b",
                    latest_tag,
                    "-f",
                    "deployment/deployment_configs.yaml:gcp_cloud_run",
                ],
                check=True,
            )

            subprocess.run(["bentoctl", "deploy", "-b", latest_tag], check=True)

            return {
                "status": "success",
                "deployment_type": "bentoml-gcp-cloudrun",
                "bento_tag": latest_tag,
                "message": "Check GCP Cloud Run console for endpoint URL",
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _deploy_bentoml_azure(self, **kwargs) -> Dict:
        """Deploy to Azure Functions using BentoML."""
        print("☁️ Deploying to Azure Functions...")

        # Similar implementation to AWS/GCP
        return {
            "status": "info",
            "message": "Azure deployment: Use bentoctl with azure-functions operator",
        }

    def _deploy_docker_compose(self, **kwargs) -> Dict:
        """Deploy using Docker Compose."""
        print("🐳 Deploying with Docker Compose...")

        try:
            # Check if docker-compose.yaml exists
            compose_file = Path("deployment/docker-compose.yaml")

            if not compose_file.exists():
                print("Creating docker-compose.yaml...")
                # Create from template
                self._create_docker_compose()

            # Deploy
            subprocess.run(["docker-compose", "-f", str(compose_file), "up", "-d"], check=True)

            return {
                "status": "success",
                "deployment_type": "docker-compose",
                "endpoints": {
                    "ray_serve": "http://localhost:8000",
                    "bentoml": "http://localhost:3000",
                    "ray_dashboard": "http://localhost:8265",
                },
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _create_docker_compose(self):
        """Create docker-compose.yaml from template."""
        compose_content = f"""
version: '3.8'

services:
  ray-serve:
    image: rayproject/ray:latest
    command: python src/deployment/ray_serve_online.py
    ports:
      - "8000:8000"
      - "8265:8265"
    environment:
      - MODEL_NAME={self.model_name}
      - MODEL_VERSION={self.model_version}
      - MLFLOW_TRACKING_URI={self.mlflow_uri}
    volumes:
      - ./src:/app/src
      - ./models:/app/models

  bentoml-service:
    image: bentoml/bentoml:latest
    command: bentoml serve src.deployment.bentoml_service:svc --port 3000
    ports:
      - "3000:3000"
    environment:
      - MODEL_NAME={self.model_name}
      - MODEL_VERSION={self.model_version}
      - MLFLOW_TRACKING_URI={self.mlflow_uri}
    volumes:
      - ./src:/app/src
      - ./models:/app/models
"""

        Path("deployment").mkdir(exist_ok=True)
        with open("deployment/docker-compose.yaml", "w") as f:
            f.write(compose_content)


# =============================================================================
# CLI
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Unified Deployment Manager")
    parser.add_argument(
        "deployment_type",
        type=str,
        choices=[
            "ray-serve-online",
            "ray-batch",
            "bentoml-local",
            "bentoml-aws-lambda",
            "bentoml-gcp-cloudrun",
            "bentoml-azure-functions",
            "docker-compose",
        ],
        help="Deployment type",
    )
    parser.add_argument("--model-name", type=str, default="iris-classifier")
    parser.add_argument("--model-version", type=str, default="latest")
    parser.add_argument("--mlflow-uri", type=str, default="http://127.0.0.1:5001")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-replicas", type=int, default=2)

    args = parser.parse_args()

    # Initialize manager
    manager = DeploymentManager(
        model_name=args.model_name, model_version=args.model_version, mlflow_uri=args.mlflow_uri
    )

    # Deploy
    result = manager.deploy(
        deployment_type=args.deployment_type, port=args.port, num_replicas=args.num_replicas
    )

    # Print result
    import json

    print("\n" + "=" * 70)
    print("DEPLOYMENT RESULT")
    print("=" * 70)
    print(json.dumps(result, indent=2))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()


# =============================================================================
# Example Usage
# =============================================================================
"""
# Ray Serve (Online)
python src/deployment/manager.py ray-serve-online \
    --model-name iris-classifier \
    --num-replicas 2

# Ray Batch
python src/deployment/manager.py ray-batch

# BentoML Local
python src/deployment/manager.py bentoml-local --port 3000

# AWS Lambda
python src/deployment/manager.py bentoml-aws-lambda

# GCP Cloud Run
python src/deployment/manager.py bentoml-gcp-cloudrun

# Docker Compose (All-in-one)
python src/deployment/manager.py docker-compose

# Python API
from src.deployment.manager import DeploymentManager

manager = DeploymentManager(
    model_name="iris-classifier",
    model_version="latest"
)

# Deploy to Ray Serve
result = manager.deploy("ray-serve-online", num_replicas=3)
print(result)
"""
