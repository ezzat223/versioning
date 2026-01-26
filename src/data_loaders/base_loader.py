"""
Base data loader class with common functionality.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd


class BaseDataLoader(ABC):
    """
    Abstract base class for all data loaders.

    All loaders must implement:
    - load_and_split()
    - log_to_mlflow()
    - get_data_info()
    - summary()
    """

    def __init__(
        self,
        data_path: str,
        target_column: Optional[str] = None,
        test_size: float = 0.2,
        validation_size: float = 0.0,
        random_state: int = 42,
    ):
        """
        Initialize base data loader.

        Args:
            data_path: Path to data source
            target_column: Target column name (None for unsupervised)
            test_size: Test set fraction
            validation_size: Validation set fraction
            random_state: Random seed
        """
        self.data_path = Path(data_path) if data_path else None
        self.target_column = target_column
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state

        # Determine task type
        self.is_supervised = target_column is not None

    @abstractmethod
    def load_and_split(self):
        """
        Load and split dataset.

        Must return: (X_train, X_test, y_train, y_test, X_val, y_val)
        For unsupervised: y_train, y_test, y_val should be None
        """

    @abstractmethod
    def get_data_info(self) -> dict:
        """
        Get dataset metadata for logging.

        Returns:
            Dictionary with dataset information
        """

    @abstractmethod
    def summary(self) -> str:
        """
        Get human-readable summary of the dataset.

        Returns:
            Formatted string with dataset information
        """

    @staticmethod
    def _is_classification(y: pd.Series) -> bool:
        """
        Heuristic to detect if task is classification vs regression.

        Rules:
        - Object/categorical dtype → classification
        - Fewer than 20 unique values → classification
        - Otherwise → regression
        """
        return y.dtype == "object" or y.dtype.name == "category" or y.nunique() < 20

    def get_feature_names(self) -> list:
        """Get list of feature column names."""
        # Default implementation - subclasses can override
        return []

    def get_target_name(self) -> Optional[str]:
        """Get target column name (None for unsupervised)."""
        return self.target_column
