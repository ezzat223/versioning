"""
Tag the most recent MLflow run as 'challenger'.
Used in CI/CD pipeline after training.
"""

import argparse
import os

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

import mlflow
from mlflow.tracking import MlflowClient


def tag_latest_run_as_challenger(
    experiment_name: str,
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"),
):
    """Tag the most recent run as challenger."""

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"❌ Experiment '{experiment_name}' not found")
        return

    # Get latest run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1
    )

    if not runs:
        print("❌ No runs found")
        return

    latest_run = runs[0]

    # Tag as challenger
    client.set_tag(latest_run.info.run_id, "model_alias", "challenger")
    client.set_tag(latest_run.info.run_id, "ci_pipeline", "true")

    print(f"✓ Tagged run {latest_run.info.run_id} as 'challenger'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tag latest run as challenger")
    parser.add_argument("--experiment-name", type=str, required=True, help="MLflow experiment name")
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"),
        help="MLflow tracking URI",
    )

    args = parser.parse_args()

    tag_latest_run_as_challenger(
        experiment_name=args.experiment_name, tracking_uri=args.tracking_uri
    )
