"""
Compare challenger vs champion models.
Fixed version with better error handling and clearer logic.
"""

import argparse
import json
import sys
from typing import Dict, Optional, Tuple

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

# Metric direction: True = higher is better, False = lower is better
METRIC_DIRECTIONS = {
    # Higher is better
    "accuracy": True,
    "test_accuracy": True,
    "val_accuracy": True,
    "precision": True,
    "recall": True,
    "f1_score": True,
    "f1": True,
    "roc_auc": True,
    "auc": True,
    "r2_score": True,
    "r2": True,
    "explained_variance": True,
    # Lower is better
    "loss": False,
    "val_loss": False,
    "test_loss": False,
    "rmse": False,
    "mse": False,
    "mae": False,
    "error": False,
}


def get_metric_direction(metric_name: str) -> bool:
    """
    Determine if higher is better for a metric.

    Returns:
        True if higher is better, False if lower is better
    """
    metric_lower = metric_name.lower()

    # Check exact matches first
    if metric_lower in METRIC_DIRECTIONS:
        return METRIC_DIRECTIONS[metric_lower]

    # Check substring matches
    for key, direction in METRIC_DIRECTIONS.items():
        if key in metric_lower:
            return direction

    # Default: assume higher is better (most common for ML)
    print(f"⚠️  Unknown metric direction for '{metric_name}', assuming higher is better")
    return True


class ModelComparator:
    """Compare challenger vs champion and decide on promotion."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "http://127.0.0.1:5001",
        improvement_threshold: float = 0.01,
    ):
        self.experiment_name = experiment_name
        self.improvement_threshold = improvement_threshold

        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

        # Get experiment
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if not self.experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found")

    def get_run_by_alias(self, alias: str) -> Optional[Run]:
        """Get most recent run with specific alias."""
        runs = self.client.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            filter_string=f"tags.model_alias = '{alias}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        return runs[0] if runs else None

    def get_latest_run(self) -> Optional[Run]:
        """Get the most recent run (challenger candidate)."""
        runs = self.client.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        return runs[0] if runs else None

    def extract_metrics(self, run: Run) -> Dict[str, float]:
        """Extract metrics from run."""
        return dict(run.data.metrics)

    def compare_metrics(
        self,
        champion_metrics: Dict[str, float],
        challenger_metrics: Dict[str, float],
        primary_metric: str,
    ) -> Tuple[bool, Dict]:
        """
        Compare metrics and decide on promotion.

        Returns:
            (should_promote, comparison_details)
        """
        # Validate metric exists
        if primary_metric not in challenger_metrics:
            return False, {
                "reason": f"Challenger missing primary metric: {primary_metric}",
                "champion_score": None,
                "challenger_score": None,
            }

        chall_score = challenger_metrics[primary_metric]

        # First model (no champion)
        if not champion_metrics or primary_metric not in champion_metrics:
            return True, {
                "reason": "First model - promoting as champion",
                "champion_score": None,
                "challenger_score": chall_score,
                "improvement": None,
                "pct_improvement": None,
            }

        champ_score = champion_metrics[primary_metric]

        # Determine direction
        higher_is_better = get_metric_direction(primary_metric)

        # Calculate improvement
        if higher_is_better:
            improvement = chall_score - champ_score
            is_better = chall_score > champ_score
        else:
            improvement = champ_score - chall_score  # Inverted for "lower is better"
            is_better = chall_score < champ_score

        # Percentage improvement
        if abs(champ_score) > 1e-10:  # Avoid division by zero
            pct_improvement = improvement / abs(champ_score)
        else:
            pct_improvement = 0

        # Decision
        should_promote = is_better and (pct_improvement >= self.improvement_threshold)

        comparison = {
            "champion_score": float(champ_score),
            "challenger_score": float(chall_score),
            "improvement": float(improvement),
            "pct_improvement": float(pct_improvement),
            "metric": primary_metric,
            "higher_is_better": higher_is_better,
        }

        if should_promote:
            comparison["reason"] = (
                f"Challenger improved by {pct_improvement*100:.2f}% "
                f"(threshold: {self.improvement_threshold*100:.2f}%)"
            )
        elif is_better:
            comparison["reason"] = (
                f"Improvement {pct_improvement*100:.2f}% below threshold "
                f"{self.improvement_threshold*100:.2f}%"
            )
        else:
            comparison["reason"] = f"Challenger performed worse on {primary_metric}"

        return should_promote, comparison

    def run_comparison(self, primary_metric: str = "test_accuracy") -> Dict:
        """Run full comparison pipeline."""
        print("\n" + "=" * 70)
        print("MODEL COMPARISON: CHAMPION vs CHALLENGER")
        print("=" * 70)

        # Get runs
        print("\nFetching runs...")
        champion_run = self.get_run_by_alias("champion")
        challenger_run = self.get_latest_run()

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
        if champion_metrics:
            print(f"  Champion:   {primary_metric} = {champion_metrics.get(primary_metric, 'N/A')}")
        print(f"  Challenger: {primary_metric} = {challenger_metrics.get(primary_metric, 'N/A')}")

        # Compare
        should_promote, comparison = self.compare_metrics(
            champion_metrics, challenger_metrics, primary_metric
        )

        # Build report
        report = {
            "experiment_name": self.experiment_name,
            "champion_run_id": champion_run.info.run_id if champion_run else None,
            "challenger_run_id": challenger_run.info.run_id,
            "comparison": comparison,
            "promote_challenger": should_promote,
            "all_challenger_metrics": challenger_metrics,
        }

        # Print decision
        print("\n" + "-" * 70)
        print("DECISION:")
        print("-" * 70)
        if should_promote:
            print("✅ PROMOTE CHALLENGER")
        else:
            print("⚠️  KEEP CHAMPION")
        print(f"   {comparison['reason']}")
        print("=" * 70 + "\n")

        return report


def main():
    parser = argparse.ArgumentParser(description="Compare challenger vs champion")
    parser.add_argument("--experiment-name", required=True, help="MLflow experiment name")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5001", help="MLflow URI")
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=0.01,
        help="Min improvement (default: 0.01 = 1%%)",
    )
    parser.add_argument("--metric", default="test_accuracy", help="Primary metric to compare")
    parser.add_argument("--output", default="comparison_report.json", help="Output file")

    args = parser.parse_args()

    try:
        comparator = ModelComparator(
            experiment_name=args.experiment_name,
            tracking_uri=args.tracking_uri,
            improvement_threshold=args.improvement_threshold,
        )

        report = comparator.run_comparison(primary_metric=args.metric)

        # Save report
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)

        print(f"✓ Report saved to {args.output}")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Comparison failed: {e}")

        # Save error report
        error_report = {"error": str(e), "promote_challenger": False}
        with open(args.output, "w") as f:
            json.dump(error_report, f, indent=2)

        sys.exit(1)


if __name__ == "__main__":
    main()
