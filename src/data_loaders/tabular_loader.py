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
        file_format: Optional[str] = None,
        auto_log_mlflow: bool = True
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
            auto_log_mlflow: Automatically log datasets to MLflow on load_and_split
        """
        super().__init__(
            data_path=data_path,
            target_column=target_column,
            test_size=test_size,
            validation_size=validation_size,
            random_state=random_state
        )
        
        self.file_format = file_format or self._detect_format()
        self.auto_log_mlflow = auto_log_mlflow
    
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
        """Load and split tabular data with automatic MLflow logging."""
        df = self._load_data()
        
        if self.is_supervised:
            result = self._split_supervised(df)
        else:
            result = self._split_unsupervised(df)
        
        # Automatic MLflow logging
        if self.auto_log_mlflow:
            self._log_splits_to_mlflow(result)
        
        return result
    
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
    
    def _log_splits_to_mlflow(self, splits: Tuple) -> None:
        """
        Log train/test/validation splits to MLflow as separate datasets.
        
        Args:
            splits: Tuple of (X_train, X_test, y_train, y_test, X_val, y_val)
        """
        try:
            X_train, X_test, y_train, y_test, X_val, y_val = splits
            
            source_path = str(self.data_path.absolute())
            dataset_name = self.data_path.stem
            
            # Log training dataset
            if self.is_supervised:
                train_df = X_train.copy()
                train_df[self.target_column] = y_train.values
                train_dataset = mlflow.data.from_pandas(
                    train_df,
                    source=source_path,
                    targets=self.target_column,
                    name=f"{dataset_name}-train"
                )
            else:
                train_dataset = mlflow.data.from_pandas(
                    X_train,
                    source=source_path,
                    name=f"{dataset_name}-train"
                )
            
            mlflow.log_input(train_dataset, context="training")
            
            # Log test dataset
            if self.is_supervised:
                test_df = X_test.copy()
                test_df[self.target_column] = y_test.values
                test_dataset = mlflow.data.from_pandas(
                    test_df,
                    source=source_path,
                    targets=self.target_column,
                    name=f"{dataset_name}-test"
                )
            else:
                test_dataset = mlflow.data.from_pandas(
                    X_test,
                    source=source_path,
                    name=f"{dataset_name}-test"
                )
            
            mlflow.log_input(test_dataset, context="testing")
            
            # Log validation dataset if exists
            if X_val is not None:
                if self.is_supervised:
                    val_df = X_val.copy()
                    val_df[self.target_column] = y_val.values
                    val_dataset = mlflow.data.from_pandas(
                        val_df,
                        source=source_path,
                        targets=self.target_column,
                        name=f"{dataset_name}-validation"
                    )
                else:
                    val_dataset = mlflow.data.from_pandas(
                        X_val,
                        source=source_path,
                        name=f"{dataset_name}-validation"
                    )
                
                mlflow.log_input(val_dataset, context="validation")
                print(f"✓ Logged train, validation, and test datasets to MLflow")
            else:
                print(f"✓ Logged train and test datasets to MLflow")
            
            # Log dataset metadata
            info = self.get_data_info()
            for key, value in info.items():
                mlflow.set_tag(key, str(value))
                
        except Exception as e:
            print(f"⚠️  MLflow logging failed: {e}")
            print("   Continuing without MLflow logging...")
    
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
