"""
Promote challenger model to champion and archive old champion.
"""

import argparse
import json
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient


class ModelPromoter:
    """Handle model promotion in MLflow."""

    def __init__(
        self,
        experiment_name: str,
        model_name: str = None,
        tracking_uri: str = "http://127.0.0.1:5001",
    ):
        """
        Initialize model promoter.

        Args:
            experiment_name: MLflow experiment name
            model_name: Registered model name (optional)
            tracking_uri: MLflow tracking server URI
        """
        self.experiment_name = experiment_name
        self.model_name = model_name or experiment_name.replace("-", "_")

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
        Promote challenger to champion and archive old champion.

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

        # Archive old champion
        if champion:
            print(f"Old Champion: {champion.info.run_id}")
            print("  → Archiving old champion...")

            self.client.set_tag(champion.info.run_id, "model_alias", "archived")
            self.client.set_tag(champion.info.run_id, "archived_at", datetime.now().isoformat())
            self.client.set_tag(champion.info.run_id, "archived_reason", "Replaced by better model")

            promotion_details["archived_run_id"] = champion.info.run_id
            print("  ✓ Old champion archived")
        else:
            print("No existing champion (first promotion)")

        # Promote challenger to champion
        print("\n→ Promoting challenger to champion...")
        self.client.set_tag(challenger.info.run_id, "model_alias", "champion")
        self.client.set_tag(challenger.info.run_id, "promoted_at", datetime.now().isoformat())
        self.client.set_tag(challenger.info.run_id, "promoted_from", "challenger")

        print("✓ Challenger promoted to champion")

        # Register model in MLflow Model Registry (if not exists)
        print(f"\n→ Registering model '{self.model_name}'...")
        try:
            # Get model URI
            model_uri = f"runs:/{challenger.info.run_id}/model"

            # Register model
            model_version = mlflow.register_model(model_uri, self.model_name)

            # Set version alias to "champion"
            self.client.set_registered_model_alias(
                self.model_name, "champion", model_version.version
            )

            # Archive old champion version if exists
            if champion:
                # Find old champion version
                versions = self.client.search_model_versions(f"run_id='{champion.info.run_id}'")
                if versions:
                    old_version = versions[0].version
                    self.client.set_registered_model_alias(self.model_name, "archived", old_version)

            promotion_details["model_name"] = self.model_name
            promotion_details["model_version"] = model_version.version

            print(f"✓ Model registered as version {model_version.version}")
            print(f"✓ Alias 'champion' set to version {model_version.version}")

        except Exception as e:
            print(f"⚠️  Model registration failed: {e}")
            print("   (This is OK if model registry is not configured)")

        # Extract metrics for reporting
        if challenger.data.metrics:
            promotion_details["metrics"] = dict(challenger.data.metrics)

        print("\n" + "=" * 70)
        print("✅ PROMOTION COMPLETE")
        print("=" * 70)
        print(f"\nNew Champion: {challenger.info.run_id}")
        if "model_version" in promotion_details:
            print(f"Model Version: {promotion_details['model_version']}")
        print()

        return promotion_details


def main():
    parser = argparse.ArgumentParser(description="Promote challenger model to champion")
    parser.add_argument("--experiment-name", type=str, required=True, help="MLflow experiment name")
    parser.add_argument(
        "--model-name", type=str, help="Registered model name (defaults to experiment name)"
    )
    parser.add_argument(
        "--tracking-uri", type=str, default="http://127.0.0.1:5001", help="MLflow tracking URI"
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
