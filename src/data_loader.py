"""
Generic data loading module for supervised and unsupervised learning.
Automatically detects task type based on whether target column is provided.
"""
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split


class DataLoader:
    """
    Generic data loader for ML projects supporting both supervised and unsupervised learning.
    
    Examples:
        # Supervised learning (classification/regression)
        loader = DataLoader("data.csv", target_column="label")
        X_train, X_test, y_train, y_test, _, _ = loader.load_and_split()
        
        # Unsupervised learning (clustering/dimensionality reduction)
        loader = DataLoader("data.csv", target_column=None)
        X_train, X_test, _, _, _, _ = loader.load_and_split()
    """
    
    def __init__(
        self, 
        data_path: str,
        target_column: Optional[str] = None,
        test_size: float = 0.2,
        validation_size: float = 0.0,
        random_state: int = 42
    ):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to dataset CSV (must be DVC-tracked for versioning)
            target_column: Name of target column (None for unsupervised learning)
            test_size: Fraction for test set
            validation_size: Fraction for validation set (from training data)
            random_state: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.target_column = target_column
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        
        # Determine task type
        self.is_supervised = target_column is not None
        
        # Validate file exists
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}\n"
                f"If DVC-tracked, run: dvc pull"
            )
    
    def load_and_split(
        self
    ) -> Tuple[
        pd.DataFrame, 
        pd.DataFrame, 
        Optional[pd.Series], 
        Optional[pd.Series], 
        Optional[pd.DataFrame], 
        Optional[pd.Series]
    ]:
        """
        Load and split dataset into train/val/test.
        
        Returns:
            For supervised learning:
                (X_train, X_test, y_train, y_test, X_val, y_val)
            
            For unsupervised learning:
                (X_train, X_test, None, None, X_val, None)
        """
        # Load data
        df = pd.read_csv(self.data_path)
        
        if self.is_supervised:
            return self._split_supervised(df)
        else:
            return self._split_unsupervised(df)
    
    def _split_supervised(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
        """Split data for supervised learning."""
        
        # Validate target column exists
        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in dataset.\n"
                f"Available columns: {df.columns.tolist()}"
            )
        
        # Separate features and target
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]
        
        # Determine if classification (for stratification)
        stratify = y if self._is_classification(y) else None
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify
        )
        
        # Optional validation split
        X_val, y_val = None, None
        if self.validation_size > 0:
            val_size_adjusted = self.validation_size / (1 - self.test_size)
            stratify_val = y_train if self._is_classification(y_train) else None
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=val_size_adjusted,
                random_state=self.random_state,
                stratify=stratify_val
            )
        
        return X_train, X_test, y_train, y_test, X_val, y_val
    
    def _split_unsupervised(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, None, None, Optional[pd.DataFrame], None]:
        """Split data for unsupervised learning."""
        
        X = df
        
        # Train-test split (no stratification for unsupervised)
        X_train, X_test = train_test_split(
            X,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # Optional validation split
        X_val = None
        if self.validation_size > 0:
            val_size_adjusted = self.validation_size / (1 - self.test_size)
            X_train, X_val = train_test_split(
                X_train,
                test_size=val_size_adjusted,
                random_state=self.random_state
            )
        
        return X_train, X_test, None, None, X_val, None
    
    def get_data_info(self) -> dict:
        """Get dataset information for logging."""
        df = pd.read_csv(self.data_path)
        
        info = {
            "data.path": str(self.data_path.absolute()),
            "data.rows": len(df),
            "data.columns": len(df.columns),
            "data.task_type": "supervised" if self.is_supervised else "unsupervised",
            "data.test_size": self.test_size,
            "data.validation_size": self.validation_size,
            "data.random_state": self.random_state,
        }
        
        # Add supervised-specific info
        if self.is_supervised:
            info["data.target"] = self.target_column
            y = df[self.target_column]
            info["data.target_type"] = "classification" if self._is_classification(y) else "regression"
            
            if self._is_classification(y):
                info["data.num_classes"] = y.nunique()
                info["data.class_distribution"] = str(y.value_counts().to_dict())
        
        return info
    
    @staticmethod
    def _is_classification(y: pd.Series) -> bool:
        """
        Heuristic to detect if task is classification vs regression.
        
        Rules:
        - Object/categorical dtype → classification
        - Fewer than 20 unique values → classification
        - Otherwise → regression
        """
        return y.dtype == 'object' or y.dtype.name == 'category' or y.nunique() < 20
    
    def get_feature_names(self) -> list:
        """Get list of feature column names."""
        df = pd.read_csv(self.data_path)
        if self.is_supervised:
            return [col for col in df.columns if col != self.target_column]
        return df.columns.tolist()
    
    def get_target_name(self) -> Optional[str]:
        """Get target column name (None for unsupervised)."""
        return self.target_column
    
    def summary(self) -> str:
        """Get human-readable summary of the dataset."""
        # df = pd.read_csv(self.data_path)
        info = self.get_data_info()
        
        summary_lines = [
            "="*60,
            "DATASET SUMMARY",
            "="*60,
            f"Path: {self.data_path}",
            f"Task Type: {info['data.task_type'].upper()}",
            f"Rows: {info['data.rows']:,}",
            f"Columns: {info['data.columns']}",
            "",
            f"Split Configuration:",
            f"  - Train: {(1 - self.test_size - self.validation_size)*100:.1f}%",
            f"  - Test: {self.test_size*100:.1f}%",
        ]
        
        if self.validation_size > 0:
            summary_lines.append(f"  - Validation: {self.validation_size*100:.1f}%")
        
        if self.is_supervised:
            summary_lines.extend([
                "",
                f"Target Column: {self.target_column}",
                f"Target Type: {info['data.target_type']}",
            ])
            
            if info['data.target_type'] == 'classification':
                summary_lines.append(f"Number of Classes: {info['data.num_classes']}")
        else:
            summary_lines.extend([
                "",
                "No target column (unsupervised learning)",
                "All columns treated as features",
            ])
        
        summary_lines.append("="*60)
        
        return "\n".join(summary_lines)


class UnsupervisedDataLoader(DataLoader):
    """
    Convenience class for unsupervised learning.
    Explicitly sets target_column=None.
    """
    
    def __init__(
        self,
        data_path: str,
        test_size: float = 0.2,
        validation_size: float = 0.0,
        random_state: int = 42
    ):
        """Initialize unsupervised data loader."""
        super().__init__(
            data_path=data_path,
            target_column=None,
            test_size=test_size,
            validation_size=validation_size,
            random_state=random_state
        )


class SupervisedDataLoader(DataLoader):
    """
    Convenience class for supervised learning.
    Requires target_column to be specified.
    """
    
    def __init__(
        self,
        data_path: str,
        target_column: str,
        test_size: float = 0.2,
        validation_size: float = 0.0,
        random_state: int = 42
    ):
        """Initialize supervised data loader."""
        if target_column is None:
            raise ValueError(
                "SupervisedDataLoader requires target_column to be specified.\n"
                "Use UnsupervisedDataLoader if no target column."
            )
        
        super().__init__(
            data_path=data_path,
            target_column=target_column,
            test_size=test_size,
            validation_size=validation_size,
            random_state=random_state
        )
