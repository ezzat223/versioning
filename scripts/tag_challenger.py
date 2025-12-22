"""
Tag the most recent MLflow run as 'challenger'.
Used in CI/CD pipeline after training.
"""

import mlflow
from mlflow.tracking import MlflowClient


def tag_latest_run_as_challenger(
    experiment_name: str = "iris-classification-ci", tracking_uri: str = "http://127.0.0.1:5001"
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
    tag_latest_run_as_challenger()
