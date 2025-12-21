"""
Tabular data loader for CSV/Parquet files.
Supports supervised and unsupervised learning with DVC tracking.
"""
import pandas as pd
import mlflow
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

from .base_loader import BaseDataLoader


class TabularDataLoader(BaseDataLoader):
    """
    Data loader for tabular data (CSV, Parquet, Excel).
    
    Examples:
        # CSV file
        loader = TabularDataLoader("data/dataset.csv", target_column="label")
        
        # Parquet file
        loader = TabularDataLoader("data/dataset.parquet", target_column="price")
        
        # Unsupervised
        loader = TabularDataLoader("data/features.csv", target_column=None)
    """
    
    def __init__(
        self,
        data_path: str,
        target_column: Optional[str] = None,
        test_size: float = 0.2,
        validation_size: float = 0.0,
        random_state: int = 42,
        file_format: Optional[str] = None
    ):
        """
        Initialize tabular data loader.
        
        Args:
            data_path: Path to data file (CSV, Parquet, Excel)
            target_column: Target column name (None for unsupervised)
            test_size: Test set fraction
            validation_size: Validation set fraction
            random_state: Random seed
            file_format: Force format ('csv', 'parquet', 'excel'). Auto-detect if None.
        """
        super().__init__(
            data_path=data_path,
            target_column=target_column,
            test_size=test_size,
            validation_size=validation_size,
            random_state=random_state
        )
        
        self.file_format = file_format or self._detect_format()
    
    def _detect_format(self) -> str:
        """Detect file format from extension."""
        suffix = self.data_path.suffix.lower()
        
        format_map = {
            '.csv': 'csv',
            '.parquet': 'parquet',
            '.pq': 'parquet',
            '.xlsx': 'excel',
            '.xls': 'excel',
        }
        
        if suffix in format_map:
            return format_map[suffix]
        else:
            raise ValueError(
                f"Unknown file format: {suffix}\n"
                f"Supported: .csv, .parquet, .pq, .xlsx, .xls\n"
                f"Or specify file_format explicitly."
            )
    
    def _load_data(self) -> pd.DataFrame:
        """Load data from file."""
        if self.file_format == 'csv':
            return pd.read_csv(self.data_path)
        elif self.file_format == 'parquet':
            return pd.read_parquet(self.data_path)
        elif self.file_format == 'excel':
            return pd.read_excel(self.data_path)
        else:
            raise ValueError(f"Unsupported format: {self.file_format}")
    
    def load_and_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series], 
                                      Optional[pd.Series], Optional[pd.DataFrame], Optional[pd.Series]]:
        """Load and split tabular data."""
        df = self._load_data()
        
        if self.is_supervised:
            return self._split_supervised(df)
        else:
            return self._split_unsupervised(df)
    
    def _split_supervised(self, df: pd.DataFrame) -> Tuple:
        """Split for supervised learning."""
        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found.\n"
                f"Available: {df.columns.tolist()}"
            )
        
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]
        
        stratify = y if self._is_classification(y) else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify
        )
        
        X_val, y_val = None, None
        if self.validation_size > 0:
            val_size = self.validation_size / (1 - self.test_size)
            stratify_val = y_train if self._is_classification(y_train) else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=self.random_state, stratify=stratify_val
            )
        
        return X_train, X_test, y_train, y_test, X_val, y_val
    
    def _split_unsupervised(self, df: pd.DataFrame) -> Tuple:
        """Split for unsupervised learning."""
        X = df
        
        X_train, X_test = train_test_split(
            X, test_size=self.test_size, random_state=self.random_state
        )
        
        X_val = None
        if self.validation_size > 0:
            val_size = self.validation_size / (1 - self.test_size)
            X_train, X_val = train_test_split(
                X_train, test_size=val_size, random_state=self.random_state
            )
        
        return X_train, X_test, None, None, X_val, None
    
    def log_to_mlflow(self, context: str = "training") -> mlflow.data.dataset.Dataset:
        """
        Log tabular dataset to MLflow with proper source information.
        
        Args:
            context: Context for the dataset (e.g., 'training', 'validation')
        
        Returns:
            MLflow Dataset object
        """
        df = self._load_data()
        
        # Create MLflow dataset with proper source
        dataset = mlflow.data.from_pandas(
            df,
            source=str(self.data_path.absolute()),
            targets=self.target_column,
            name=self.data_path.stem  # Use filename as dataset name
        )
        
        # Log to current MLflow run
        mlflow.log_input(dataset, context=context)
        
        # Log additional metadata
        mlflow.set_tag("data.loader_type", "tabular")
        mlflow.set_tag("data.file_format", self.file_format)
        mlflow.set_tag("data.source_type", "local_file")
        
        return dataset
    
    def get_data_info(self) -> dict:
        """Get dataset metadata."""
        df = self._load_data()
        
        info = {
            "data.loader_type": "tabular",
            "data.file_format": self.file_format,
            "data.path": str(self.data_path.absolute()),
            "data.rows": len(df),
            "data.columns": len(df.columns),
            "data.memory_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "data.task_type": "supervised" if self.is_supervised else "unsupervised",
            "data.test_size": self.test_size,
            "data.validation_size": self.validation_size,
            "data.random_state": self.random_state,
        }
        
        if self.is_supervised:
            y = df[self.target_column]
            info["data.target"] = self.target_column
            info["data.target_type"] = "classification" if self._is_classification(y) else "regression"
            
            if self._is_classification(y):
                info["data.num_classes"] = y.nunique()
        
        return info
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        info = self.get_data_info()
        
        lines = [
            "="*60,
            "TABULAR DATASET SUMMARY",
            "="*60,
            f"File: {self.data_path.name}",
            f"Format: {self.file_format.upper()}",
            f"Size: {info['data.memory_mb']:.2f} MB",
            f"Rows: {info['data.rows']:,}",
            f"Columns: {info['data.columns']}",
            "",
            f"Task: {info['data.task_type'].upper()}",
        ]
        
        if self.is_supervised:
            lines.extend([
                f"Target: {self.target_column}",
                f"Type: {info['data.target_type']}",
            ])
            if info['data.target_type'] == 'classification':
                lines.append(f"Classes: {info['data.num_classes']}")
        
        lines.append("="*60)
        return "\n".join(lines)
