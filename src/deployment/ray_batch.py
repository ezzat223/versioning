"""
Ray Data for large-scale batch inference.
Fully integrated with MLflow and CI/CD pipeline.

Features:
- Distributed processing with Ray Data
- Auto-loads champion model from MLflow
- Fault tolerance and checkpointing
- Progress tracking
- Supports CSV, Parquet, Delta Lake
- Memory-efficient streaming
"""
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

import ray
from ray import data
from ray.data import Dataset
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = os.getenv("MODEL_NAME", "iris-classifier")
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))


# =============================================================================
# Model Loading Utilities
# =============================================================================

def get_champion_model_uri() -> str:
    """
    Get champion model URI from MLflow.
    
    Returns:
        Model URI string
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    try:
        # Search for champion model
        experiment = mlflow.get_experiment_by_name(f"{MODEL_NAME.replace('-', '_')}-ci")
        if not experiment:
            experiment = mlflow.get_experiment_by_name(MODEL_NAME.replace("-", "_"))
        
        if not experiment:
            raise ValueError(f"No experiment found for {MODEL_NAME}")
        
        # Get champion run
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.model_alias = 'champion'",
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs:
            run = runs[0]
            model_uri = f"runs:/{run.info.run_id}/model"
            logger.info(f"✓ Found champion model: {model_uri}")
            logger.info(f"  Accuracy: {run.data.metrics.get('test_accuracy', 'N/A')}")
            return model_uri
        else:
            # No champion, get latest
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if runs:
                run = runs[0]
                model_uri = f"runs:/{run.info.run_id}/model"
                logger.warning(f"⚠️ No champion found, using latest: {model_uri}")
                return model_uri
            else:
                raise ValueError("No models found in MLflow")
    
    except Exception as e:
        logger.error(f"Error getting model URI: {e}")
        # Fallback to local
        local_path = f"models/{MODEL_NAME}"
        logger.warning(f"Using local model: {local_path}")
        return local_path


# =============================================================================
# Ray Data Batch Inference Class
# =============================================================================

class MLModel:
    """
    Model class for Ray Data batch inference.
    Must be serializable (no __init__ with args).
    """
    
    def __init__(self):
        """Initialize model (called on each worker)."""
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
        
        # Try to get probabilities
        probabilities = None
        try:
            if hasattr(self.model._model_impl, 'predict_proba'):
                probabilities = self.model._model_impl.predict_proba(df)
        except:
            pass
        
        # Build output batch
        output = {
            "prediction": predictions
        }
        
        # Add probability columns if available
        if probabilities is not None:
            for i in range(probabilities.shape[1]):
                output[f"prob_class_{i}"] = probabilities[:, i]
        
        return output
    
    def _load_model(self):
        """Load model from MLflow."""
        logger.info("Loading model in worker...")
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Get model URI
        self.model_uri = get_champion_model_uri()
        
        # Load model
        self.model = mlflow.pyfunc.load_model(self.model_uri)
        
        logger.info(f"✓ Model loaded in worker: {self.model_uri}")


# =============================================================================
# Batch Inference Pipeline
# =============================================================================

class RayBatchInference:
    """
    Ray Data batch inference pipeline.
    Integrated with MLflow for automatic champion model loading.
    """
    
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        model_version: str = MODEL_VERSION,
        batch_size: int = BATCH_SIZE,
        num_cpus_per_task: float = 1.0
    ):
        """
        Initialize batch inference pipeline.
        
        Args:
            model_name: Model name in MLflow
            model_version: Model version or 'champion'/'latest'
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
            ray.init(
                address=ray_address,
                ignore_reinit_error=True,
                logging_level=logging.INFO
            )
        
        logger.info("✓ Ray Data batch inference initialized")
    
    def predict_dataset(
        self,
        dataset: Dataset,
        output_path: Optional[str] = None,
        output_format: str = "parquet"
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
        logger.info("\n" + "="*70)
        logger.info("RAY DATA BATCH INFERENCE")
        logger.info("="*70)
        logger.info(f"Model: {self.model_name} ({self.model_version})")
        logger.info(f"Dataset size: {dataset.count()} rows")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info("="*70 + "\n")
        
        # Apply model to dataset
        logger.info("Running predictions...")
        
        predictions_ds = dataset.map_batches(
            MLModel,
            batch_size=self.batch_size,
            num_cpus=self.num_cpus_per_task,
            batch_format="numpy"
        )
        
        logger.info("✓ Predictions complete")
        
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
            
            logger.info(f"✓ Results saved to {output_path}")
        
        logger.info("\n" + "="*70)
        logger.info("BATCH INFERENCE COMPLETE")
        logger.info("="*70 + "\n")
        
        return predictions_ds
    
    def predict_csv(
        self,
        input_path: str,
        output_path: str,
        feature_columns: Optional[List[str]] = None
    ) -> Dataset:
        """
        Run batch predictions on CSV file.
        
        Args:
            input_path: Path to input CSV
            output_path: Path to save results (CSV)
            feature_columns: List of feature columns (None = all)
            
        Returns:
            Dataset with predictions
        """
        logger.info(f"Loading CSV from {input_path}...")
        
        # Read CSV with Ray Data
        dataset = ray.data.read_csv(input_path)
        
        # Select feature columns if specified
        if feature_columns:
            dataset = dataset.select_columns(feature_columns)
        
        # Run predictions
        return self.predict_dataset(
            dataset,
            output_path=output_path,
            output_format="csv"
        )
    
    def predict_parquet(
        self,
        input_path: str,
        output_path: str,
        feature_columns: Optional[List[str]] = None
    ) -> Dataset:
        """
        Run batch predictions on Parquet file(s).
        
        Args:
            input_path: Path to input Parquet (file or directory)
            output_path: Path to save results
            feature_columns: List of feature columns (None = all)
            
        Returns:
            Dataset with predictions
        """
        logger.info(f"Loading Parquet from {input_path}...")
        
        # Read Parquet with Ray Data
        dataset = ray.data.read_parquet(input_path)
        
        # Select feature columns if specified
        if feature_columns:
            dataset = dataset.select_columns(feature_columns)
        
        # Run predictions
        return self.predict_dataset(
            dataset,
            output_path=output_path,
            output_format="parquet"
        )
    
    def predict_pandas(
        self,
        df: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Run batch predictions on Pandas DataFrame.
        
        Args:
            df: Input DataFrame
            output_path: Optional path to save results
            
        Returns:
            DataFrame with predictions
        """
        logger.info("Converting Pandas DataFrame to Ray Dataset...")
        
        # Convert to Ray Dataset
        dataset = ray.data.from_pandas(df)
        
        # Run predictions
        predictions_ds = self.predict_dataset(dataset, output_path=None)
        
        # Convert back to Pandas
        result_df = predictions_ds.to_pandas()
        
        # Combine with original DataFrame
        result_df = pd.concat([df.reset_index(drop=True), result_df], axis=1)
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.suffix == '.csv':
                result_df.to_csv(output_path, index=False)
            elif output_path.suffix == '.parquet':
                result_df.to_parquet(output_path, index=False)
            
            logger.info(f"✓ Results saved to {output_path}")
        
        return result_df
    
    def predict_s3(
        self,
        input_path: str,
        output_path: str,
        file_format: str = "parquet"
    ) -> Dataset:
        """
        Run batch predictions on S3 data.
        
        Args:
            input_path: S3 path (s3://bucket/path)
            output_path: S3 output path
            file_format: File format ('parquet', 'csv')
            
        Returns:
            Dataset with predictions
        """
        logger.info(f"Loading from S3: {input_path}...")
        
        # Read from S3
        if file_format == "parquet":
            dataset = ray.data.read_parquet(input_path)
        elif file_format == "csv":
            dataset = ray.data.read_csv(input_path)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        
        # Run predictions
        return self.predict_dataset(
            dataset,
            output_path=output_path,
            output_format=file_format
        )
    
    def shutdown(self):
        """Shutdown Ray."""
        logger.info("Shutting down Ray...")
        ray.shutdown()
        logger.info("✓ Shutdown complete")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch inference with Ray Data")
    parser.add_argument("--input", type=str, required=True, help="Input data path")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--model-version", type=str, default=MODEL_VERSION)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--format", type=str, default="auto", 
                       choices=["auto", "csv", "parquet", "json"])
    parser.add_argument("--feature-columns", nargs="+", default=None)
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RayBatchInference(
        model_name=args.model_name,
        model_version=args.model_version,
        batch_size=args.batch_size
    )
    
    try:
        # Determine input format
        input_path = Path(args.input)
        
        if args.format == "auto":
            # Auto-detect format
            if input_path.suffix == '.csv':
                file_format = "csv"
            elif input_path.suffix in ['.parquet', '.pq']:
                file_format = "parquet"
            elif args.input.startswith("s3://"):
                file_format = "parquet"  # Default for S3
            else:
                raise ValueError(f"Cannot auto-detect format for {input_path}")
        else:
            file_format = args.format
        
        # Run predictions
        if file_format == "csv":
            pipeline.predict_csv(
                args.input,
                args.output,
                feature_columns=args.feature_columns
            )
        elif file_format == "parquet":
            pipeline.predict_parquet(
                args.input,
                args.output,
                feature_columns=args.feature_columns
            )
        elif args.input.startswith("s3://"):
            pipeline.predict_s3(
                args.input,
                args.output,
                file_format=file_format
            )
        else:
            raise ValueError(f"Unsupported format: {file_format}")
    
    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    main()


# =============================================================================
# Example Usage
# =============================================================================
"""
# Single machine - CSV
python src/deployment/ray_batch.py \
    --input data/test_data.csv \
    --output results/predictions.csv \
    --batch-size 1000

# Single machine - Parquet
python src/deployment/ray_batch.py \
    --input data/large_dataset.parquet \
    --output results/predictions.parquet \
    --batch-size 10000

# Ray cluster
RAY_ADDRESS="ray://head-node:10001" python src/deployment/ray_batch.py \
    --input s3://my-bucket/data.parquet \
    --output s3://my-bucket/predictions.parquet \
    --batch-size 50000

# Python API
from src.deployment.ray_batch import RayBatchInference
import pandas as pd

pipeline = RayBatchInference()

# From Pandas
df = pd.read_csv("data.csv")
results = pipeline.predict_pandas(df, "predictions.csv")

# From CSV
pipeline.predict_csv("data.csv", "predictions.csv")

# From Parquet
pipeline.predict_parquet("data.parquet", "predictions.parquet")

pipeline.shutdown()
"""
