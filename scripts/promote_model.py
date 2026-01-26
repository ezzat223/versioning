"""
Promote challenger model to champion and archive old champion.
Uses MLflow Model Registry aliases for deployment.
"""

import argparse
import json
import os
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

import mlflow
from mlflow.tracking import MlflowClient


class ModelPromoter:
    """Handle model promotion using MLflow aliases."""

    def __init__(
        self,
        experiment_name: str,
        model_name: str = None,
        tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"),
    ):
        """
        Initialize model promoter.

        Args:
            experiment_name: MLflow experiment name
            model_name: Registered model name (optional, defaults to experiment_name)
            tracking_uri: MLflow tracking server URI
        """
        self.experiment_name = experiment_name
        self.model_name = model_name or experiment_name.replace("-exp", "-model")

        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

        # Get experiment
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if not self.experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found")

    def get_run_by_alias(self, alias: str) -> mlflow.entities.Run:
        """Get run by model alias tag."""
        runs = self.client.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            filter_string=f"tags.model_alias = '{alias}'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        return runs[0] if runs else None

    def promote(self) -> dict:
        """
        Promote challenger to champion using MLflow aliases.

        Uses BOTH run tags and Model Registry aliases for compatibility.

        Returns:
            Promotion details dictionary
        """
        print("\n" + "=" * 70)
        print("MODEL PROMOTION")
        print("=" * 70)

        # Get challenger
        challenger = self.get_run_by_alias("challenger")
        if not challenger:
            raise ValueError("No challenger found")

        print(f"\nChallenger: {challenger.info.run_id}")

        # Get current champion
        champion = self.get_run_by_alias("champion")

        promotion_details = {
            "challenger_run_id": challenger.info.run_id,
            "champion_run_id": champion.info.run_id if champion else None,
            "promotion_time": datetime.now().isoformat(),
            "experiment_name": self.experiment_name,
        }

        # Archive old champion (run tag)
        if champion:
            print(f"Old Champion: {champion.info.run_id}")
            print("  → Archiving old champion...")

            self.client.set_tag(champion.info.run_id, "model_alias", "archived")
            self.client.set_tag(champion.info.run_id, "archived_at", datetime.now().isoformat())
            self.client.set_tag(champion.info.run_id, "archived_reason", "Replaced by better model")

            promotion_details["archived_run_id"] = champion.info.run_id
            print("  ✓ Old champion archived (run tag)")
        else:
            print("No existing champion (first promotion)")

        # Promote challenger to champion (run tag)
        print("\n→ Promoting challenger to champion...")
        self.client.set_tag(challenger.info.run_id, "model_alias", "champion")
        self.client.set_tag(challenger.info.run_id, "promoted_at", datetime.now().isoformat())
        self.client.set_tag(challenger.info.run_id, "promoted_from", "challenger")

        print("✓ Challenger promoted to champion (run tag)")

        # Register model in MLflow Model Registry with aliases
        print("\n→ Updating Model Registry aliases...")
        try:
            # Get model URI
            model_uri = f"runs:/{challenger.info.run_id}/model"

            # Register model (or get existing)
            try:
                model_version = mlflow.register_model(model_uri, self.model_name)
                print(f"  ✓ Model registered as version {model_version.version}")
            except Exception as e:
                # Model might already be registered
                versions = self.client.search_model_versions(f"run_id='{challenger.info.run_id}'")
                if versions:
                    model_version = versions[0]
                    print(f"  ✓ Using existing version {model_version.version}")
                else:
                    raise e

            # Set 'champion' alias on new version
            self.client.set_registered_model_alias(
                self.model_name, "champion", model_version.version
            )
            print(f"  ✓ Set alias 'champion' → version {model_version.version}")

            # Archive old champion in registry
            if champion:
                # Find old champion version
                old_versions = self.client.search_model_versions(f"run_id='{champion.info.run_id}'")
                if old_versions:
                    old_version = old_versions[0].version
                    # Set 'archived' alias on old version
                    self.client.set_registered_model_alias(self.model_name, "archived", old_version)
                    print(f"  ✓ Set alias 'archived' → version {old_version}")

            promotion_details["model_name"] = self.model_name
            promotion_details["model_version"] = model_version.version

        except Exception as e:
            print(f"⚠️  Model Registry update failed: {e}")
            print("   (Promotion succeeded with run tags, but Registry aliases not updated)")

        # Extract metrics for reporting
        if challenger.data.metrics:
            promotion_details["metrics"] = dict(challenger.data.metrics)

        print("\n" + "=" * 70)
        print("✅ PROMOTION COMPLETE")
        print("=" * 70)
        print(f"\nNew Champion: {challenger.info.run_id}")
        if "model_version" in promotion_details:
            print(f"Model Version: {promotion_details['model_version']} (alias: @champion)")
        print()

        return promotion_details


def main():
    parser = argparse.ArgumentParser(description="Promote challenger model to champion")
    parser.add_argument("--experiment-name", type=str, required=True, help="MLflow experiment name")
    parser.add_argument(
        "--model-name", type=str, help="Registered model name (defaults to experiment name)"
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"),
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="promotion_details.json",
        help="Output file for promotion details",
    )

    args = parser.parse_args()

    # Promote model
    promoter = ModelPromoter(
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        tracking_uri=args.tracking_uri,
    )

    details = promoter.promote()

    # Save details
    with open(args.output, "w") as f:
        json.dump(details, f, indent=2)

    print(f"✓ Promotion details saved to {args.output}")


if __name__ == "__main__":
    main()
