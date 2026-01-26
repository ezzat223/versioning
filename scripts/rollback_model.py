"""
Rollback to previous champion model.
Used for manual recovery in case of issues.
"""

import argparse
import os
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

import mlflow
from mlflow.tracking import MlflowClient


def rollback_to_previous_champion(
    experiment_name: str,
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"),
):
    """
    Rollback current champion to previous archived model.

    Args:
        experiment_name: MLflow experiment name
        tracking_uri: MLflow tracking URI
    """
    print("\n" + "=" * 70)
    print("MODEL ROLLBACK")
    print("=" * 70)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    # Get current champion
    champion_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.model_alias = 'champion'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not champion_runs:
        print("❌ No current champion found")
        return

    current_champion = champion_runs[0]
    print(f"\nCurrent Champion: {current_champion.info.run_id}")

    # Get most recent archived model
    archived_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.model_alias = 'archived'",
        order_by=["tags.archived_at DESC"],
        max_results=1,
    )

    if not archived_runs:
        print("❌ No archived model found for rollback")
        return

    previous_champion = archived_runs[0]
    print(f"Previous Champion: {previous_champion.info.run_id}")

    # Confirm rollback
    print("\n⚠️  This will:")
    print(f"  1. Archive current champion: {current_champion.info.run_id}")
    print(f"  2. Restore previous champion: {previous_champion.info.run_id}")

    # Archive current champion
    print("\n→ Archiving current champion...")
    client.set_tag(current_champion.info.run_id, "model_alias", "archived")
    client.set_tag(current_champion.info.run_id, "archived_at", datetime.now().isoformat())
    client.set_tag(
        current_champion.info.run_id, "archived_reason", "Rollback - replaced by previous model"
    )
    print("✓ Current champion archived")

    # Restore previous champion
    print("\n→ Restoring previous champion...")
    client.set_tag(previous_champion.info.run_id, "model_alias", "champion")
    client.set_tag(previous_champion.info.run_id, "restored_at", datetime.now().isoformat())
    client.delete_tag(previous_champion.info.run_id, "archived_at")
    print("✓ Previous champion restored")

    print("\n" + "=" * 70)
    print("✅ ROLLBACK COMPLETE")
    print("=" * 70)
    print(f"\nNew Champion: {previous_champion.info.run_id}")
    print("\nMetrics:")
    for metric, value in previous_champion.data.metrics.items():
        print(f"  {metric}: {value:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Rollback to previous champion model")
    parser.add_argument("--experiment-name", type=str, required=True, help="MLflow experiment name")
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"),
        help="MLflow tracking URI",
    )

    args = parser.parse_args()

    rollback_to_previous_champion(
        experiment_name=args.experiment_name, tracking_uri=args.tracking_uri
    )


if __name__ == "__main__":
    main()
