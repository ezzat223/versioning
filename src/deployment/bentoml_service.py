"""
BentoML service for serverless deployments.
Supports: AWS Lambda, GCP Cloud Functions, Azure Functions

Features:
- Automatic batching for efficiency
- Cold start optimization
- Multi-cloud deployment
- API versioning
- Swagger/OpenAPI docs
"""

import os
from typing import Dict, List

import bentoml
import mlflow
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = os.getenv("MODEL_NAME", "iris-classifier")
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
SERVICE_NAME = f"{MODEL_NAME}-service"


# =============================================================================
# Pydantic Models
# =============================================================================


class PredictionInput(BaseModel):
    """Input schema for predictions."""

    features: List[List[float]] = Field(
        ..., description="2D array of feature values", example=[[5.1, 3.5, 1.4, 0.2]]
    )
    feature_names: List[str] = Field(
        default=None,
        description="Optional feature names",
        example=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    )


class PredictionOutput(BaseModel):
    """Output schema for predictions."""

    predictions: List[int]
    probabilities: List[List[float]] = None
    model_name: str
    model_version: str


# =============================================================================
# Model Loading and Saving
# =============================================================================


def load_model_from_mlflow():
    """Load model from MLflow and save to BentoML."""
    print(f"Loading model from MLflow: {MODEL_NAME} (v{MODEL_VERSION})")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        # Load model
        if MODEL_VERSION == "latest":
            model_uri = f"models:/{MODEL_NAME}@champion"
        else:
            model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

        model = mlflow.pyfunc.load_model(model_uri)
        print(f"✓ Model loaded from {model_uri}")

        # Save to BentoML
        bento_model = bentoml.mlflow.save_model(
            name=MODEL_NAME,
            model=model,
            signatures={"predict": {"batchable": True, "batch_dim": 0}},
            metadata={"model_version": MODEL_VERSION, "mlflow_uri": model_uri},
        )

        print(f"✓ Model saved to BentoML: {bento_model.tag}")
        return bento_model

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise


# =============================================================================
# BentoML Service Definition
# =============================================================================

# Load or get model
try:
    # Try to get existing model
    model_ref = bentoml.mlflow.get(f"{MODEL_NAME}:latest")
    print(f"✓ Using existing BentoML model: {model_ref.tag}")
except bentoml.exceptions.NotFound:
    # Load from MLflow if not exists
    model_ref = load_model_from_mlflow()

# Create service
svc = bentoml.Service(
    name=SERVICE_NAME, runners=[bentoml.mlflow.get(f"{MODEL_NAME}:latest").to_runner()]
)


@svc.api(
    input=bentoml.io.JSON(pydantic_model=PredictionInput),
    output=bentoml.io.JSON(pydantic_model=PredictionOutput),
    route="/predict",
)
async def predict(input_data: PredictionInput) -> PredictionOutput:
    """
    Predict endpoint with automatic batching.

    Args:
        input_data: Prediction input with features

    Returns:
        Predictions with probabilities
    """
    # Convert to DataFrame
    if input_data.feature_names:
        df = pd.DataFrame(input_data.features, columns=input_data.feature_names)
    else:
        df = pd.DataFrame(input_data.features)

    # Get runner
    runner = svc.runners[0]

    # Make predictions
    predictions = await runner.predict.async_run(df)

    # Get probabilities if available
    probabilities = None
    try:
        probabilities = await runner.predict_proba.async_run(df)
        probabilities = probabilities.tolist()
    except:
        pass

    return PredictionOutput(
        predictions=predictions.tolist() if hasattr(predictions, "tolist") else predictions,
        probabilities=probabilities,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
    )


@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON(), route="/health")
def health() -> Dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "service": SERVICE_NAME,
    }


@svc.api(input=bentoml.io.JSON(), output=bentoml.io.JSON(), route="/info")
def info() -> Dict:
    """Service information endpoint."""
    return {
        "service_name": SERVICE_NAME,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "endpoints": {"predict": "/predict", "health": "/health", "info": "/info"},
    }


# =============================================================================
# Deployment Configuration
# =============================================================================


def create_bentofile():
    """Create bentofile.yaml for deployment."""
    bentofile_content = f"""
service: "src.deployment.bentoml_service:svc"
description: "ML Model Inference Service - {MODEL_NAME}"
labels:
  owner: ml-team
  project: mlops-pipeline
include:
  - "src/deployment/bentoml_service.py"
  - "src/utils.py"
python:
  packages:
    - mlflow==2.10.0
    - pandas==2.1.3
    - scikit-learn==1.3.2
    - xgboost==2.0.3
    - numpy==1.26.2
docker:
  distro: debian
  python_version: "3.11"
  system_packages:
    - curl
  setup_script: |
    apt-get update && apt-get install -y --no-install-recommends curl
"""

    with open("bentofile.yaml", "w") as f:
        f.write(bentofile_content)

    print("✓ Created bentofile.yaml")


# =============================================================================
# CLI Commands
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BentoML Service Manager")
    parser.add_argument(
        "command", choices=["load", "serve", "build", "containerize"], help="Command to execute"
    )
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--model-version", type=str, default=MODEL_VERSION)
    parser.add_argument("--port", type=int, default=3000)

    args = parser.parse_args()

    # Update environment
    os.environ["MODEL_NAME"] = args.model_name
    os.environ["MODEL_VERSION"] = args.model_version

    if args.command == "load":
        # Load model from MLflow to BentoML
        load_model_from_mlflow()
        print("\n✓ Model loaded to BentoML")
        print(f"  View models: bentoml models list")

    elif args.command == "serve":
        # Serve locally for testing
        print(f"\n🚀 Starting BentoML server on port {args.port}...")
        print(f"   API docs: http://localhost:{args.port}/docs")
        import subprocess

        subprocess.run(
            ["bentoml", "serve", "src.deployment.bentoml_service:svc", "--port", str(args.port)]
        )

    elif args.command == "build":
        # Build Bento
        create_bentofile()
        print("\n📦 Building Bento...")
        import subprocess

        subprocess.run(["bentoml", "build"])
        print("\n✓ Bento built successfully")
        print("  View bentos: bentoml list")

    elif args.command == "containerize":
        # Containerize
        create_bentofile()
        print("\n🐳 Containerizing Bento...")
        import subprocess

        subprocess.run(["bentoml", "build"])
        # Get latest bento tag
        result = subprocess.run(
            ["bentoml", "list", "--output", "json"], capture_output=True, text=True
        )
        import json

        bentos = json.loads(result.stdout)
        if bentos:
            latest_tag = bentos[0]["tag"]
            print(f"\n📦 Containerizing {latest_tag}...")
            subprocess.run(
                ["bentoml", "containerize", latest_tag, "-t", f"{args.model_name}:latest"]
            )
            print(f"\n✓ Docker image created: {args.model_name}:latest")


# =============================================================================
# Example Usage
# =============================================================================
"""
# 1. Load model from MLflow to BentoML
python src/deployment/bentoml_service.py load \
    --model-name iris-classifier \
    --model-version latest

# 2. Serve locally for testing
python src/deployment/bentoml_service.py serve --port 3000

# Test API
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[5.1, 3.5, 1.4, 0.2]],
    "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
  }'

# 3. Build Bento for deployment
python src/deployment/bentoml_service.py build

# 4. Containerize for Docker deployment
python src/deployment/bentoml_service.py containerize

# 5. Deploy to AWS Lambda (requires bentoctl)
bentoctl build -b <bento_tag> -f deployment_config_lambda.yaml
bentoctl deploy -b <bento_tag>

# 6. Deploy to GCP Cloud Run
bentoctl build -b <bento_tag> -f deployment_config_gcp.yaml
bentoctl deploy -b <bento_tag>

# 7. Deploy to Azure Functions
bentoctl build -b <bento_tag> -f deployment_config_azure.yaml
bentoctl deploy -b <bento_tag>
"""
