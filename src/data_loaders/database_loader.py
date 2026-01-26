"""
Database data loader for various external databases.
Supports PostgreSQL, MySQL, MongoDB, BigQuery, Snowflake, etc.
User provides configured client and table/collection name.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from .base_loader import BaseDataLoader


class DatabaseDataLoader(BaseDataLoader):
    """
    Data loader for external databases.

    Supported databases (via user-provided clients):
    - PostgreSQL / MySQL (via sqlalchemy)
    - MongoDB (via pymongo)
    - BigQuery (via google-cloud-bigquery)
    - Snowflake (via snowflake-connector-python)
    - Redis (via redis-py)
    - Cassandra (via cassandra-driver)
    - Any database with pandas-compatible read methods

    Examples:
        # PostgreSQL with SQLAlchemy
        from sqlalchemy import create_engine
        engine = create_engine('postgresql://user:pass@localhost/db')
        loader = DatabaseDataLoader(
            client=engine,
            table_name="users",
            target_column="churn",
            database_type="postgresql"
        )

        # MongoDB
        from pymongo import MongoClient
        client = MongoClient('mongodb://localhost:27017/')
        db = client['mydb']
        loader = DatabaseDataLoader(
            client=db,
            table_name="customers",
            target_column="segment",
            database_type="mongodb"
        )

        # BigQuery
        from google.cloud import bigquery
        client = bigquery.Client()
        loader = DatabaseDataLoader(
            client=client,
            table_name="project.dataset.table",
            target_column="label",
            database_type="bigquery"
        )

        # Custom SQL query
        loader = DatabaseDataLoader(
            client=engine,
            query="SELECT * FROM users WHERE active=true",
            target_column="category",
            database_type="postgresql"
        )
    """

    def __init__(
        self,
        client: Any,
        table_name: Optional[str] = None,
        query: Optional[str] = None,
        target_column: Optional[str] = None,
        database_type: str = "unknown",
        test_size: float = 0.2,
        validation_size: float = 0.0,
        random_state: int = 42,
        cache_data: bool = True,
        cache_path: str = ".cache/database_cache.parquet",
        auto_log_mlflow: bool = True,
    ):
        """
        Initialize database data loader.

        Args:
            client: Database client/connection (sqlalchemy engine, pymongo db, etc.)
            table_name: Table/collection name
            query: Custom SQL/query (overrides table_name)
            target_column: Target column name (None for unsupervised)
            database_type: Type of database for logging
            test_size: Test set fraction
            validation_size: Validation set fraction
            random_state: Random seed
            cache_data: Cache data locally for reproducibility
            cache_path: Path to cache file
            auto_log_mlflow: Automatically log datasets to MLflow on load_and_split
        """
        # Initialize without data_path (will be set from cache)
        self.client = client
        self.table_name = table_name
        self.query = query
        self.database_type = database_type
        self.cache_data = cache_data
        self.cache_path = Path(cache_path)
        self.auto_log_mlflow = auto_log_mlflow

        # Validate inputs
        if table_name is None and query is None:
            raise ValueError("Must provide either table_name or query")

        # Load data and cache if needed
        self._cached_data = None

        # Set data_path to cache location
        data_path = str(self.cache_path) if cache_data else "database://memory"

        super().__init__(
            data_path=data_path,
            target_column=target_column,
            test_size=test_size,
            validation_size=validation_size,
            random_state=random_state,
        )

    def _detect_database_type(self) -> str:
        """Auto-detect database type from client."""
        client_type = type(self.client).__name__

        type_map = {
            "Engine": "sqlalchemy",
            "MongoClient": "mongodb",
            "Database": "mongodb",
            "Client": "bigquery",
            "SnowflakeConnection": "snowflake",
            "Redis": "redis",
            "Cluster": "cassandra",
        }

        for key, value in type_map.items():
            if key in client_type:
                return value

        return "unknown"

    def _load_data(self) -> pd.DataFrame:
        """Load data from database with caching."""
        # Check cache first
        if self.cache_data and self.cache_path.exists():
            print(f"Loading cached data from {self.cache_path}")
            return pd.read_parquet(self.cache_path)

        print(f"Fetching data from {self.database_type} database...")

        # Load based on database type
        if hasattr(self.client, "execute"):
            # SQLAlchemy or similar
            df = self._load_from_sql()
        elif "pymongo" in str(type(self.client).__module__):
            # MongoDB
            df = self._load_from_mongodb()
        elif "bigquery" in str(type(self.client).__module__):
            # BigQuery
            df = self._load_from_bigquery()
        elif hasattr(self.client, "cursor"):
            # Generic database with cursor
            df = self._load_from_cursor()
        else:
            raise ValueError(
                f"Unsupported client type: {type(self.client)}\n"
                "Provide a client with execute() or cursor() method, "
                "or implement custom loading logic."
            )

        print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")

        # Cache data
        if self.cache_data:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.cache_path)
            print(f"✓ Cached data to {self.cache_path}")

        return df

    def _load_from_sql(self) -> pd.DataFrame:
        """Load from SQL database (PostgreSQL, MySQL, etc.)."""
        if self.query:
            return pd.read_sql(self.query, self.client)
        else:
            return pd.read_sql_table(self.table_name, self.client)

    def _load_from_mongodb(self) -> pd.DataFrame:
        """Load from MongoDB."""
        if self.query:
            # Query is a MongoDB query dict
            cursor = self.client[self.table_name].find(
                eval(self.query) if isinstance(self.query, str) else self.query
            )
        else:
            cursor = self.client[self.table_name].find()

        data = list(cursor)
        return pd.DataFrame(data)

    def _load_from_bigquery(self) -> pd.DataFrame:
        """Load from BigQuery."""
        if self.query:
            return self.client.query(self.query).to_dataframe()
        else:
            query = f"SELECT * FROM `{self.table_name}`"
            return self.client.query(query).to_dataframe()

    def _load_from_cursor(self) -> pd.DataFrame:
        """Load from generic database with cursor."""
        cursor = self.client.cursor()

        if self.query:
            cursor.execute(self.query)
        else:
            cursor.execute(f"SELECT * FROM {self.table_name}")

        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()

        return pd.DataFrame(data, columns=columns)

    def load_and_split(
        self,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        Optional[pd.Series],
        Optional[pd.Series],
        Optional[pd.DataFrame],
        Optional[pd.Series],
    ]:
        """Load and split database data with automatic MLflow logging."""
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
                X_train,
                y_train,
                test_size=val_size,
                random_state=self.random_state,
                stratify=stratify_val,
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

            # Create source string
            if self.query:
                source = f"{self.database_type}://custom_query_{self._compute_query_hash()}"
            else:
                source = f"{self.database_type}://{self.table_name}"

            dataset_name = self.table_name or f"query_{self._compute_query_hash()[:8]}"

            # Log training dataset
            if self.is_supervised:
                train_df = X_train.copy()
                train_df[self.target_column] = y_train.values
                train_dataset = mlflow.data.from_pandas(
                    train_df,
                    source=source,
                    targets=self.target_column,
                    name=f"{dataset_name}-train",
                )
            else:
                train_dataset = mlflow.data.from_pandas(
                    X_train, source=source, name=f"{dataset_name}-train"
                )

            mlflow.log_input(train_dataset, context="training")

            # Log test dataset
            if self.is_supervised:
                test_df = X_test.copy()
                test_df[self.target_column] = y_test.values
                test_dataset = mlflow.data.from_pandas(
                    test_df, source=source, targets=self.target_column, name=f"{dataset_name}-test"
                )
            else:
                test_dataset = mlflow.data.from_pandas(
                    X_test, source=source, name=f"{dataset_name}-test"
                )

            mlflow.log_input(test_dataset, context="testing")

            # Log validation dataset if exists
            if X_val is not None:
                if self.is_supervised:
                    val_df = X_val.copy()
                    val_df[self.target_column] = y_val.values
                    val_dataset = mlflow.data.from_pandas(
                        val_df,
                        source=source,
                        targets=self.target_column,
                        name=f"{dataset_name}-validation",
                    )
                else:
                    val_dataset = mlflow.data.from_pandas(
                        X_val, source=source, name=f"{dataset_name}-validation"
                    )

                mlflow.log_input(val_dataset, context="validation")
                print("✓ Logged train, validation, and test datasets to MLflow")
            else:
                print("✓ Logged train and test datasets to MLflow")

            # Log dataset metadata
            info = self.get_data_info()
            for key, value in info.items():
                mlflow.set_tag(key, str(value))

        except Exception as e:
            print(f"⚠️  MLflow logging failed: {e}")
            print("   Continuing without MLflow logging...")

    def _compute_query_hash(self) -> str:
        """Compute hash of query/table for versioning."""
        query_str = self.query if self.query else f"SELECT * FROM {self.table_name}"
        return hashlib.sha256(query_str.encode()).hexdigest()[:16]

    def get_data_info(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        df = self._load_data()

        info = {
            "data.loader_type": "database",
            "data.database_type": self.database_type,
            "data.table_name": self.table_name or "custom_query",
            "data.rows": len(df),
            "data.columns": len(df.columns),
            "data.memory_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "data.task_type": "supervised" if self.is_supervised else "unsupervised",
            "data.test_size": self.test_size,
            "data.validation_size": self.validation_size,
            "data.random_state": self.random_state,
            "data.cached": self.cache_data,
        }

        if self.query:
            info["data.query_hash"] = self._compute_query_hash()

        if self.is_supervised:
            y = df[self.target_column]
            info["data.target"] = self.target_column
            info["data.target_type"] = (
                "classification" if self._is_classification(y) else "regression"
            )

            if self._is_classification(y):
                info["data.num_classes"] = y.nunique()

        return info

    def summary(self) -> str:
        """Generate human-readable summary."""
        info = self.get_data_info()

        lines = [
            "=" * 60,
            "DATABASE DATASET SUMMARY",
            "=" * 60,
            f"Database: {self.database_type.upper()}",
            f"Source: {self.table_name or 'Custom Query'}",
            f"Rows: {info['data.rows']:,}",
            f"Columns: {info['data.columns']}",
            f"Size: {info['data.memory_mb']:.2f} MB",
            "",
            f"Task: {info['data.task_type'].upper()}",
        ]

        if self.is_supervised:
            lines.extend(
                [
                    f"Target: {self.target_column}",
                    f"Type: {info['data.target_type']}",
                ]
            )
            if info["data.target_type"] == "classification":
                lines.append(f"Classes: {info['data.num_classes']}")

        if self.cache_data:
            lines.extend(
                [
                    "",
                    f"Cached: {self.cache_path}",
                ]
            )

        lines.append("=" * 60)
        return "\n".join(lines)
