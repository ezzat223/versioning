"""
Data loading and versioning using MLflow Datasets API.
"""
import hashlib
import pandas as pd
from pathlib import Path
from typing import Tuple
import mlflow
# from mlflow.data.pandas_dataset import PandasDataset
from sklearn.model_selection import train_test_split


class IrisDataLoader:
    """
    Manages Iris dataset loading and versioning with MLflow Datasets.
    """
    
    def __init__(self, data_path: str = "data/iris.csv", test_size: float = 0.2, random_state: int = 42):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to iris.csv file
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.test_size = test_size
        self.random_state = random_state
        
        # Ensure data directory exists
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
    
    def compute_dataset_hash(self) -> str:
        """
        Compute MD5 hash of the dataset file for versioning.
        
        Returns:
            MD5 hash as hexadecimal string
        """
        md5_hash = hashlib.md5()
        with open(self.data_path, "rb") as f:
            # Read file in chunks for memory efficiency
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load and split the Iris dataset.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Load data
        df = pd.read_csv(self.data_path)
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def log_to_mlflow(self) -> mlflow.data.dataset.Dataset:
        """
        Log dataset to MLflow using the Datasets API.
        
        Returns:
            MLflow Dataset object
        """
        # Load full dataset
        df = pd.read_csv(self.data_path)
        
        # Compute dataset version (hash)
        dataset_hash = self.compute_dataset_hash()
        
        # Create MLflow PandasDataset
        dataset = mlflow.data.from_pandas(
            df,
            source=str(self.data_path.absolute()),
            name="iris-dataset",
            targets="target"
        )
        
        # Log dataset to current MLflow run
        mlflow.log_input(dataset, context="training")
        
        # Log dataset metadata as tags
        mlflow.set_tag("dataset.name", "iris-dataset")
        mlflow.set_tag("dataset.version", dataset_hash[:8])  # Short hash
        mlflow.set_tag("dataset.hash", dataset_hash)  # Full hash
        mlflow.set_tag("dataset.path", str(self.data_path))
        mlflow.set_tag("dataset.rows", len(df))
        mlflow.set_tag("dataset.columns", len(df.columns))
        
        print("✓ Dataset logged to MLflow")
        print("  - Name: iris-dataset")
        print(f"  - Version: {dataset_hash[:8]}")
        print(f"  - Rows: {len(df)}")
        
        return dataset
    
    def get_metadata(self) -> dict:
        """
        Get dataset metadata for logging.
        
        Returns:
            Dictionary with dataset metadata
        """
        df = pd.read_csv(self.data_path)
        dataset_hash = self.compute_dataset_hash()
        
        return {
            "dataset.name": "iris-dataset",
            "dataset.version": dataset_hash[:8],
            "dataset.hash": dataset_hash,
            "dataset.path": str(self.data_path.absolute()),
            "dataset.rows": len(df),
            "dataset.columns": len(df.columns),
            "dataset.test_size": self.test_size,
            "dataset.random_state": self.random_state,
        }
