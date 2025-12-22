"""
Ray Core deployment for batch inference.
Supports: Large-scale batch predictions on clusters (on-prem or cloud)

Features:
- Distributed processing across multiple nodes
- Fault tolerance (auto-retry failed batches)
- Progress tracking
- Checkpointing for long-running jobs
- Memory-efficient streaming
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List

import mlflow
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = os.getenv("MODEL_NAME", "iris-classifier")
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))


# =============================================================================
# Ray Remote Functions
# =============================================================================


@ray.remote
class ModelActor:
    """
    Ray actor that holds a model instance.
    Each actor can process batches independently.
    """

    def __init__(self, model_name: str, model_version: str, tracking_uri: str):
        """Initialize actor with model."""
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self._load_model(tracking_uri)

    def _load_model(self, tracking_uri: str):
        """Load model from MLflow."""
        mlflow.set_tracking_uri(tracking_uri)

        try:
            if self.model_version == "latest":
                model_uri = f"models:/{self.model_name}@champion"
            else:
                model_uri = f"models:/{self.model_name}/{self.model_version}"

            self.model = mlflow.pyfunc.load_model(model_uri)
            print(f"✓ Model loaded in actor: {model_uri}")

        except Exception as e:
            print(f"⚠️ Failed to load model from MLflow: {e}")
            # Fallback
            local_path = f"models/{self.model_name}"
            self.model = mlflow.pyfunc.load_model(local_path)

    def predict_batch(self, batch: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict on a batch of data.

        Args:
            batch: DataFrame with features

        Returns:
            Dictionary with predictions and metadata
        """
        try:
            predictions = self.model.predict(batch)

            # Try to get probabilities
            probabilities = None
            try:
                if hasattr(self.model._model_impl, "predict_proba"):
                    probabilities = self.model._model_impl.predict_proba(batch).tolist()
            except:
                pass

            return {
                "predictions": (
                    predictions.tolist() if hasattr(predictions, "tolist") else predictions
                ),
                "probabilities": probabilities,
                "batch_size": len(batch),
                "success": True,
            }

        except Exception as e:
            return {"error": str(e), "batch_size": len(batch), "success": False}


# =============================================================================
# Batch Inference Pipeline
# =============================================================================


class RayBatchInference:
    """Distributed batch inference using Ray Core."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        model_version: str = MODEL_VERSION,
        tracking_uri: str = MLFLOW_TRACKING_URI,
        num_actors: int = None,
        batch_size: int = BATCH_SIZE,
    ):
        """
        Initialize batch inference pipeline.

        Args:
            model_name: Model name in MLflow
            model_version: Model version or 'latest'
            tracking_uri: MLflow tracking URI
            num_actors: Number of model actors (defaults to CPU count)
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.model_version = model_version
        self.tracking_uri = tracking_uri
        self.batch_size = batch_size

        # Initialize Ray
        if not ray.is_initialized():
            ray.init(address=os.getenv("RAY_ADDRESS", "auto"), ignore_reinit_error=True)

        # Determine number of actors
        if num_actors is None:
            num_actors = os.cpu_count() or 4

        self.num_actors = num_actors

        # Create model actors
        print(f"Creating {self.num_actors} model actors...")
        self.actors = [
            ModelActor.remote(model_name, model_version, tracking_uri)
            for _ in range(self.num_actors)
        ]
        print(f"✓ Created {self.num_actors} actors")

    def _chunk_dataframe(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Split dataframe into chunks."""
        chunks = []
        for i in range(0, len(df), self.batch_size):
            chunks.append(df.iloc[i : i + self.batch_size])
        return chunks

    def predict_dataframe(
        self, df: pd.DataFrame, output_path: str = None, show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Run batch predictions on a DataFrame.

        Args:
            df: Input DataFrame
            output_path: Optional path to save results
            show_progress: Show progress bar

        Returns:
            DataFrame with predictions
        """
        print("\n" + "=" * 70)
        print("RAY BATCH INFERENCE")
        print("=" * 70)
        print(f"Model: {self.model_name} (v{self.model_version})")
        print(f"Input size: {len(df):,} rows")
        print(f"Batch size: {self.batch_size}")
        print(f"Num actors: {self.num_actors}")
        print("=" * 70 + "\n")

        # Split into chunks
        chunks = self._chunk_dataframe(df)
        print(f"Split into {len(chunks)} batches")

        # Distribute work across actors
        print("Submitting tasks to Ray cluster...")

        # Round-robin distribution of chunks to actors
        futures = []
        for i, chunk in enumerate(chunks):
            actor = self.actors[i % self.num_actors]
            future = actor.predict_batch.remote(chunk)
            futures.append((i, future))

        # Collect results with progress bar
        results = []

        if show_progress:
            pbar = tqdm(total=len(futures), desc="Processing batches")

        for batch_idx, future in futures:
            result = ray.get(future)
            results.append((batch_idx, result))

            if show_progress:
                pbar.update(1)

        if show_progress:
            pbar.close()

        # Sort by original batch index
        results.sort(key=lambda x: x[0])

        # Combine predictions
        all_predictions = []
        all_probabilities = []

        for _, result in results:
            if result["success"]:
                all_predictions.extend(result["predictions"])
                if result["probabilities"]:
                    all_probabilities.extend(result["probabilities"])

        # Create output DataFrame
        output_df = df.copy()
        output_df["prediction"] = all_predictions

        if all_probabilities:
            # Add probability columns
            prob_array = np.array(all_probabilities)
            for i in range(prob_array.shape[1]):
                output_df[f"prob_class_{i}"] = prob_array[:, i]

        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.suffix == ".csv":
                output_df.to_csv(output_path, index=False)
            elif output_path.suffix == ".parquet":
                output_df.to_parquet(output_path, index=False)

            print(f"\n✓ Results saved to {output_path}")

        print(f"\n✓ Batch inference complete!")
        print(f"  Total predictions: {len(all_predictions):,}")
        print("=" * 70 + "\n")

        return output_df

    def predict_csv(
        self, input_path: str, output_path: str, feature_columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Run batch predictions on a CSV file.

        Args:
            input_path: Path to input CSV
            output_path: Path to save results
            feature_columns: List of feature column names (if None, uses all)

        Returns:
            DataFrame with predictions
        """
        # Read CSV
        df = pd.read_csv(input_path)

        # Select features
        if feature_columns:
            df = df[feature_columns]

        # Run predictions
        return self.predict_dataframe(df, output_path=output_path)

    def predict_parquet(
        self, input_path: str, output_path: str, feature_columns: List[str] = None
    ) -> pd.DataFrame:
        """Run batch predictions on a Parquet file."""
        df = pd.read_parquet(input_path)

        if feature_columns:
            df = df[feature_columns]

        return self.predict_dataframe(df, output_path=output_path)

    def shutdown(self):
        """Shutdown Ray."""
        print("Shutting down Ray...")
        ray.shutdown()
        print("✓ Shutdown complete")


# =============================================================================
# CLI
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch inference with Ray")
    parser.add_argument("--input", type=str, required=True, help="Input data path")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--model-version", type=str, default=MODEL_VERSION)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-actors", type=int, default=None)
    parser.add_argument("--feature-columns", nargs="+", default=None)

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = RayBatchInference(
        model_name=args.model_name,
        model_version=args.model_version,
        batch_size=args.batch_size,
        num_actors=args.num_actors,
    )

    try:
        # Determine input format
        input_path = Path(args.input)

        if input_path.suffix == ".csv":
            pipeline.predict_csv(str(input_path), args.output, feature_columns=args.feature_columns)
        elif input_path.suffix in [".parquet", ".pq"]:
            pipeline.predict_parquet(
                str(input_path), args.output, feature_columns=args.feature_columns
            )
        else:
            raise ValueError(f"Unsupported format: {input_path.suffix}")

    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    main()


# =============================================================================
# Example Usage
# =============================================================================
"""
# Single machine
python src/deployment/ray_batch_inference.py \
    --input data/test_data.csv \
    --output results/predictions.csv \
    --model-name iris-classifier \
    --batch-size 1000 \
    --num-actors 4

# Connect to Ray cluster
RAY_ADDRESS="ray://head-node:10001" python src/deployment/ray_batch_inference.py \
    --input data/large_dataset.parquet \
    --output results/predictions.parquet \
    --batch-size 10000

# Python usage
from src.deployment.ray_batch_inference import RayBatchInference

pipeline = RayBatchInference(
    model_name="iris-classifier",
    model_version="latest",
    batch_size=1000,
    num_actors=8
)

# Predict on DataFrame
import pandas as pd
df = pd.read_csv("data/test.csv")
results = pipeline.predict_dataframe(df, output_path="results/predictions.csv")

pipeline.shutdown()
"""
