"""
Compare challenger model against champion model.
Determines if challenger should be promoted based on performance metrics.
"""

import argparse
import json
import sys
from typing import Dict, Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient


class ModelComparator:
    """Compare two models and decide on promotion."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "http://127.0.0.1:5001",
        improvement_threshold: float = 0.01,  # 1% improvement required
    ):
        """
        Initialize model comparator.

        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking server URI
            improvement_threshold: Minimum improvement required (e.g., 0.01 = 1%)
        """
        self.experiment_name = experiment_name
        self.improvement_threshold = improvement_threshold

        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

        # Get experiment
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if not self.experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found")

    def get_champion_run(self) -> Optional[mlflow.entities.Run]:
        """Get current champion model run."""
        runs = self.client.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            filter_string="tags.model_alias = 'champion'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        return runs[0] if runs else None

    def get_challenger_run(self) -> Optional[mlflow.entities.Run]:
        """Get latest challenger model run."""
        runs = self.client.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            filter_string="tags.model_alias = 'challenger'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        return runs[0] if runs else None

    def get_latest_run(self) -> Optional[mlflow.entities.Run]:
        """Get the most recent run (to be evaluated as challenger)."""
        runs = self.client.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )

        return runs[0] if runs else None

    def extract_metrics(self, run: mlflow.entities.Run) -> Dict[str, float]:
        """Extract relevant metrics from a run."""
        metrics = {}

        # Primary metric
        if "test_accuracy" in run.data.metrics:
            metrics["test_accuracy"] = run.data.metrics["test_accuracy"]

        # Additional metrics
        for metric_name in ["val_accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            if metric_name in run.data.metrics:
                metrics[metric_name] = run.data.metrics[metric_name]

        return metrics

    def compare_metrics(
        self, champion_metrics: Dict[str, float], challenger_metrics: Dict[str, float]
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Compare metrics and decide if challenger should be promoted.

        Returns:
            (should_promote, comparison_details)
        """
        # Primary metric for comparison
        primary_metric = "test_accuracy"

        if primary_metric not in challenger_metrics:
            return False, {"error": f"Challenger missing {primary_metric}"}

        # If no champion exists, promote challenger
        if not champion_metrics:
            return True, {
                "reason": "No existing champion - promoting challenger",
                "challenger_metrics": challenger_metrics,
            }

        if primary_metric not in champion_metrics:
            return True, {
                "reason": f"Champion missing {primary_metric} - promoting challenger",
                "challenger_metrics": challenger_metrics,
            }

        # Calculate improvement
        champion_score = champion_metrics[primary_metric]
        challenger_score = challenger_metrics[primary_metric]
        improvement = challenger_score - champion_score
        improvement_pct = (improvement / champion_score) * 100 if champion_score > 0 else 0

        # Decision logic
        should_promote = improvement >= self.improvement_threshold

        comparison = {
            "champion_score": champion_score,
            "challenger_score": challenger_score,
            "improvement": improvement,
            "improvement_pct": improvement_pct,
            "threshold": self.improvement_threshold,
            "threshold_pct": self.improvement_threshold * 100,
            "should_promote": should_promote,
            "primary_metric": primary_metric,
            "all_champion_metrics": champion_metrics,
            "all_challenger_metrics": challenger_metrics,
        }

        if should_promote:
            comparison["reason"] = f"Challenger improved by {improvement_pct:.2f}%"
        else:
            comparison["reason"] = (
                f"Improvement {improvement_pct:.2f}% below threshold {self.improvement_threshold * 100:.2f}%"
            )

        return should_promote, comparison

    def run_comparison(self) -> Dict[str, any]:
        """
        Run full comparison pipeline.

        Returns:
            Comparison report dictionary
        """
        print("\n" + "=" * 70)
        print("MODEL COMPARISON: CHAMPION vs CHALLENGER")
        print("=" * 70)

        # Get runs
        print("\nFetching runs...")
        champion_run = self.get_champion_run()
        challenger_run = self.get_latest_run()  # Most recent run is challenger

        if not challenger_run:
            return {"error": "No runs found in experiment", "promote_challenger": False}

        # Tag challenger
        self.client.set_tag(challenger_run.info.run_id, "model_alias", "challenger")
        print(f"✓ Challenger: {challenger_run.info.run_id}")

        if champion_run:
            print(f"✓ Champion:   {champion_run.info.run_id}")
        else:
            print("✓ Champion:   None (first model)")

        # Extract metrics
        champion_metrics = self.extract_metrics(champion_run) if champion_run else {}
        challenger_metrics = self.extract_metrics(challenger_run)

        print("\nMetrics:")
        print(f"  Champion:   {champion_metrics}")
        print(f"  Challenger: {challenger_metrics}")

        # Compare
        should_promote, comparison = self.compare_metrics(champion_metrics, challenger_metrics)

        # Build report
        report = {
            "experiment_name": self.experiment_name,
            "champion_run_id": champion_run.info.run_id if champion_run else None,
            "challenger_run_id": challenger_run.info.run_id,
            "comparison": comparison,
            "promote_challenger": should_promote,
            "timestamp": challenger_run.info.start_time,
        }

        # Print decision
        print("\n" + "-" * 70)
        print("DECISION:")
        print("-" * 70)
        if should_promote:
            print(f"✅ PROMOTE CHALLENGER")
            print(f"   Reason: {comparison['reason']}")
        else:
            print(f"⚠️  KEEP CHAMPION")
            print(f"   Reason: {comparison['reason']}")
        print("=" * 70 + "\n")

        return report


def main():
    parser = argparse.ArgumentParser(description="Compare challenger vs champion models")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="iris-classification-ci",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--tracking-uri", type=str, default="http://127.0.0.1:5001", help="MLflow tracking URI"
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=0.01,
        help="Minimum improvement required (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_report.json",
        help="Output file for comparison report",
    )

    args = parser.parse_args()

    # Run comparison
    comparator = ModelComparator(
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        improvement_threshold=args.improvement_threshold,
    )

    report = comparator.run_comparison()

    # Save report
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"✓ Comparison report saved to {args.output}")

    # Exit with appropriate code
    sys.exit(0 if report.get("promote_challenger", False) else 1)


if __name__ == "__main__":
    main()
