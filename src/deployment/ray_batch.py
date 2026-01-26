"""
Ray Data for large-scale batch inference.
Fully integrated with MLflow - loads champion model automatically.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
import ray
from ray.data import Dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment
MODEL_NAME = os.getenv("MODEL_NAME", "template-model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "champion")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))


# =============================================================================
# Ray Data Batch Inference Class
# =============================================================================


class MLModel:
    """
    Model wrapper for Ray Data batch inference.
    Must be serializable (no __init__ with args).
    """

    def __init__(self):
        """Initialize model (called once per worker)."""
        self.model = None
        self.model_uri = None

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Process a batch of data.

        Args:
            batch: Dictionary with numpy arrays (Ray Data format)

        Returns:
            Dictionary with predictions
        """
        # Lazy load model (only once per worker)
        if self.model is None:
            self._load_model()

        # Convert batch to DataFrame
        df = pd.DataFrame(batch)

        # Make predictions
        predictions = self.model.predict(df)

        # Try to get probabilities if classifier
        probabilities = None
        try:
            if hasattr(self.model, "_model_impl") and hasattr(
                self.model._model_impl, "predict_proba"
            ):
                probabilities = self.model._model_impl.predict_proba(df)
        except Exception:
            pass

        # Build output batch
        output = {"prediction": predictions}

        # Add probability columns if available
        if probabilities is not None:
            for i in range(probabilities.shape[1]):
                output[f"prob_class_{i}"] = probabilities[:, i]

        return output

    def _load_model(self):
        """Load model from MLflow using alias."""
        logger.info("Loading model in worker...")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        try:
            # Load with alias (champion, staging, production)
            if MODEL_VERSION.lower() in ("champion", "staging", "production"):
                self.model_uri = f"models:/{MODEL_NAME}@{MODEL_VERSION.lower()}"
            elif MODEL_VERSION.lower() == "latest":
                self.model_uri = f"models:/{MODEL_NAME}/latest"
            else:
                # Load specific version number
                self.model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

            self.model = mlflow.pyfunc.load_model(self.model_uri)
            logger.info(f"✅ Model loaded: {self.model_uri}")

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            logger.error(f"💡 Ensure model is registered with alias '@{MODEL_VERSION}' in MLflow")
            raise


# =============================================================================
# Batch Inference Pipeline
# =============================================================================


class RayBatchInference:
    """Ray Data batch inference pipeline integrated with MLflow."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        model_version: str = MODEL_VERSION,
        batch_size: int = BATCH_SIZE,
        num_cpus_per_task: float = 1.0,
    ):
        """
        Initialize batch inference pipeline.

        Args:
            model_name: Model name in MLflow
            model_version: Model version or alias (champion, latest, etc.)
            batch_size: Batch size for processing
            num_cpus_per_task: CPUs allocated per inference task
        """
        self.model_name = model_name
        self.model_version = model_version
        self.batch_size = batch_size
        self.num_cpus_per_task = num_cpus_per_task

        # Initialize Ray
        if not ray.is_initialized():
            ray_address = os.getenv("RAY_ADDRESS", "auto")
            logger.info(f"Initializing Ray (address: {ray_address})...")
            ray.init(address=ray_address, ignore_reinit_error=True, logging_level=logging.INFO)

        logger.info("✅ Ray Data batch inference initialized")

    def predict_dataset(
        self, dataset: Dataset, output_path: Optional[str] = None, output_format: str = "parquet"
    ) -> Dataset:
        """
        Run batch predictions on Ray Dataset.

        Args:
            dataset: Ray Dataset with features
            output_path: Optional path to save results
            output_format: Output format ('parquet', 'csv', 'json')

        Returns:
            Dataset with predictions
        """
        logger.info("\n" + "=" * 70)
        logger.info("RAY DATA BATCH INFERENCE")
        logger.info("=" * 70)
        logger.info(f"Model: {self.model_name}@{self.model_version}")
        logger.info(f"Dataset size: {dataset.count()} rows")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info("=" * 70 + "\n")

        # Apply model to dataset
        logger.info("Running predictions...")

        predictions_ds = dataset.map_batches(
            MLModel,
            batch_size=self.batch_size,
            num_cpus=self.num_cpus_per_task,
            batch_format="numpy",
        )

        logger.info("✅ Predictions complete")

        # Save if output path provided
        if output_path:
            logger.info(f"Saving results to {output_path}...")

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_format == "parquet":
                predictions_ds.write_parquet(str(output_path))
            elif output_format == "csv":
                predictions_ds.write_csv(str(output_path))
            elif output_format == "json":
                predictions_ds.write_json(str(output_path))
            else:
                raise ValueError(f"Unsupported format: {output_format}")

            logger.info(f"✅ Results saved to {output_path}")

        return predictions_ds

    def predict_csv(
        self, input_path: str, output_path: str, feature_columns: Optional[List[str]] = None
    ) -> Dataset:
        """Run batch predictions on CSV file."""
        logger.info(f"Loading CSV from {input_path}...")
        dataset = ray.data.read_csv(input_path)

        if feature_columns:
            dataset = dataset.select_columns(feature_columns)

        return self.predict_dataset(dataset, output_path, output_format="csv")

    def predict_parquet(
        self, input_path: str, output_path: str, feature_columns: Optional[List[str]] = None
    ) -> Dataset:
        """Run batch predictions on Parquet file(s)."""
        logger.info(f"Loading Parquet from {input_path}...")
        dataset = ray.data.read_parquet(input_path)

        if feature_columns:
            dataset = dataset.select_columns(feature_columns)

        return self.predict_dataset(dataset, output_path, output_format="parquet")

    def predict_pandas(self, df: pd.DataFrame, output_path: Optional[str] = None) -> pd.DataFrame:
        """Run batch predictions on Pandas DataFrame."""
        logger.info("Converting DataFrame to Ray Dataset...")
        dataset = ray.data.from_pandas(df)

        predictions_ds = self.predict_dataset(dataset, output_path=None)
        result_df = predictions_ds.to_pandas()

        # Combine with original
        result_df = pd.concat([df.reset_index(drop=True), result_df], axis=1)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.suffix == ".csv":
                result_df.to_csv(output_path, index=False)
            elif output_path.suffix == ".parquet":
                result_df.to_parquet(output_path, index=False)

            logger.info(f"✅ Results saved to {output_path}")

        return result_df

    def shutdown(self):
        """Shutdown Ray."""
        logger.info("Shutting down Ray...")
        ray.shutdown()


# =============================================================================
# CLI
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch inference with Ray Data")
    parser.add_argument("--input", required=True, help="Input data path")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--model-version", default=MODEL_VERSION)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--format", default="auto", choices=["auto", "csv", "parquet"])
    parser.add_argument("--feature-columns", nargs="+", default=None)

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = RayBatchInference(
        model_name=args.model_name, model_version=args.model_version, batch_size=args.batch_size
    )

    try:
        # Auto-detect format
        input_path = Path(args.input)

        if args.format == "auto":
            if input_path.suffix == ".csv":
                file_format = "csv"
            elif input_path.suffix in [".parquet", ".pq"]:
                file_format = "parquet"
            else:
                raise ValueError(f"Cannot auto-detect format for {input_path}")
        else:
            file_format = args.format

        # Run predictions
        if file_format == "csv":
            pipeline.predict_csv(args.input, args.output, args.feature_columns)
        elif file_format == "parquet":
            pipeline.predict_parquet(args.input, args.output, args.feature_columns)

        logger.info("\n" + "=" * 70)
        logger.info("✅ BATCH INFERENCE COMPLETE")
        logger.info("=" * 70 + "\n")

    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    main()
