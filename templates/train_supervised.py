"""
TEMPLATE: Supervised Learning Training Script

INSTRUCTIONS FOR DATA SCIENTISTS:
1. Choose your data loader (delete the other two examples)
2. Implement your model selection logic
3. Add your model's hyperparameters to params.yaml and MLproject
4. Implement your evaluation metrics (if different from autolog defaults)
5. Delete these instructions and comments when ready

This template provides:
- Automatic MLflow logging via autolog
- Automatic dataset tracking (train/val/test splits logged by data loaders)
- Git metadata tracking
- Data versioning with DVC
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

import mlflow

# Ensure project root is on PYTHONPATH when running as a script (e.g., dvc repro)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import get_git_metadata, validate_git_state

warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("http://127.0.0.1:5001")

# =============================================================================
# MLFLOW AUTOLOGGING (Uncomment the framework you're using)
# =============================================================================
# Autologging automatically captures:
# - Model parameters (hyperparameters)
# - Training metrics
# - Model artifacts
# - Model signature
# - Input examples
#
# Comment out or delete the frameworks you're NOT using

# Generic autolog (works for most frameworks)
mlflow.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    disable=False,
    exclusive=False,  # Allow manual logging too
    silent=True,
)

# Framework-specific autolog (uncomment if you need more control)
# import mlflow.sklearn
# mlflow.sklearn.autolog(
#     log_input_examples=True,
#     log_model_signatures=True,
#     log_models=True,
#     max_tuning_runs=5
# )

# import mlflow.xgboost
# mlflow.xgboost.autolog(
#     log_input_examples=True,
#     log_model_signatures=True,
#     log_models=True
# )

# import mlflow.lightgbm
# mlflow.lightgbm.autolog(
#     log_input_examples=True,
#     log_model_signatures=True,
#     log_models=True
# )

# import mlflow.tensorflow
# mlflow.tensorflow.autolog(
#     log_input_examples=True,
#     log_model_signatures=True,
#     log_models=True
# )

# import mlflow.pytorch
# mlflow.pytorch.autolog(
#     log_input_examples=True,
#     log_model_signatures=True,
#     log_models=True
# )

# =============================================================================


def parse_args():
    """Parse command line arguments (configured in MLproject)."""
    parser = argparse.ArgumentParser()

    # Data parameters
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--target-column", type=str, required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)

    # TODO: Add your model's hyperparameters here
    # Example:
    # parser.add_argument("--learning-rate", type=float, default=0.001)
    # parser.add_argument("--n-estimators", type=int, default=100)
    # parser.add_argument("--max-depth", type=int, default=10)

    # MLflow parameters
    parser.add_argument("--experiment-name", type=str, default="my-experiment")
    parser.add_argument("--model-name", type=str, default="my-model")
    parser.add_argument("--strict-git", type=str, default="false")

    return parser.parse_args()


def load_data(args):
    """
    Load and split data using appropriate data loader.

    INSTRUCTIONS:
    1. Uncomment the loader you need (tabular, image, or database)
    2. Delete the other two examples
    3. Customize parameters as needed

    NOTE: Data loaders automatically log train/test/validation datasets to MLflow!

    Returns:
        X_train, X_test, y_train, y_test, X_val, y_val, loader
    """

    # ============================================================
    # OPTION 1: TABULAR DATA (CSV, Parquet, Excel)
    # ============================================================
    from src.data_loaders import TabularDataLoader

    loader = TabularDataLoader(
        data_path=args.data_path,
        target_column=args.target_column,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
        auto_log_mlflow=True,  # Automatic dataset logging
    )

    # This automatically logs datasets to MLflow!
    X_train, X_test, y_train, y_test, X_val, y_val = loader.load_and_split()

    # ============================================================
    # OPTION 2: IMAGE DATA
    # ============================================================
    # from src.data_loaders import ImageDataLoader
    #
    # loader = ImageDataLoader(
    #     data_path=args.data_path,
    #     structure_type="directory",  # or "csv"
    #     target_column=args.target_column,  # None if classes from folders
    #     image_size=(224, 224),
    #     test_size=args.test_size,
    #     validation_size=args.validation_size,
    #     random_state=args.random_state,
    #     auto_log_mlflow=True
    # )
    #
    # # This automatically logs datasets to MLflow!
    # X_train, X_test, y_train, y_test, X_val, y_val = loader.load_and_split()
    #
    # # TODO: Load actual images when needed for your model
    # # images_train = loader.load_images(X_train)
    # # images_test = loader.load_images(X_test)

    # ============================================================
    # OPTION 3: DATABASE
    # ============================================================
    # from sqlalchemy import create_engine
    # from src.data_loaders import DatabaseDataLoader
    #
    # engine = create_engine('postgresql://user:pass@localhost/db')
    #
    # loader = DatabaseDataLoader(
    #     client=engine,
    #     table_name="my_table",
    #     target_column=args.target_column,
    #     database_type="postgresql",
    #     cache_data=True,
    #     cache_path=".cache/my_data.parquet",
    #     test_size=args.test_size,
    #     validation_size=args.validation_size,
    #     random_state=args.random_state,
    #     auto_log_mlflow=True
    # )
    #
    # # This automatically logs datasets to MLflow!
    # X_train, X_test, y_train, y_test, X_val, y_val = loader.load_and_split()

    return X_train, X_test, y_train, y_test, X_val, y_val, loader


def train_model(X_train, y_train, args):
    """
    Train your model.

    INSTRUCTIONS:
    1. Replace this placeholder with your actual model
    2. Import your model class (e.g., XGBoost, LightGBM, PyTorch model)
    3. Use hyperparameters from args
    4. Return the trained model

    NOTE: Autolog will automatically log:
    - All hyperparameters
    - Training metrics
    - Model artifact
    - Model signature
    - Input examples
    """

    # TODO: Replace this with your model!
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=100,  # TODO: Get from args.n_estimators
        max_depth=10,  # TODO: Get from args.max_depth
        random_state=args.random_state,
    )

    print("Training model...")
    model.fit(X_train, y_train)
    print("✓ Model trained")

    return model


def evaluate_model(model, X_test, y_test, X_val=None, y_val=None):
    """
    Evaluate model and return metrics.

    INSTRUCTIONS:
    Add ONLY custom metrics not captured by autolog.
    Standard metrics (accuracy, loss, precision, recall, etc.) are already logged by autolog.

    Add custom metrics like:
    - Business-specific metrics
    - Domain-specific calculations
    - Custom scoring functions
    """

    print("Evaluating model...")

    # NOTE: Basic metrics are already logged by autolog!
    # Only add custom metrics here if needed

    # TODO: Add custom metrics not handled by autolog
    # Example:
    # from sklearn.metrics import matthews_corrcoef
    # custom_metrics = {
    #     "matthews_correlation": matthews_corrcoef(y_test, model.predict(X_test)),
    #     "custom_business_metric": calculate_custom_metric(y_test, predictions)
    # }
    # mlflow.log_metrics(custom_metrics)

    # Get basic metrics for display and saving (autolog already logged these)
    from sklearn.metrics import accuracy_score

    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    metrics = {"test_accuracy": test_accuracy}

    print("\n✓ Evaluation complete")
    print(f"  Test Accuracy: {test_accuracy:.4f}")

    if X_val is not None and y_val is not None:
        val_accuracy = accuracy_score(y_val, model.predict(X_val))
        metrics["val_accuracy"] = val_accuracy
        print(f"  Val Accuracy: {val_accuracy:.4f}")

    return metrics


def main():
    """Main training pipeline - MLOps infrastructure (don't modify)."""
    args = parse_args()

    print("\n" + "=" * 60)
    print("SUPERVISED LEARNING PIPELINE")
    print("=" * 60)

    # Git metadata (not handled by autolog)
    git_metadata = get_git_metadata()
    validate_git_state(git_metadata, strict=args.strict_git.lower() == "true")

    # MLflow setup
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run() as run:
        print(f"\n✓ MLflow Run ID: {run.info.run_id}")
        print("✓ Autolog enabled - automatic logging of params, metrics, and model")

        # Log git metadata (not handled by autolog)
        print("\nLogging git metadata...")
        for key, value in git_metadata.items():
            mlflow.set_tag(key, value)
        print("✓ Git metadata logged")

        # Log task type (not handled by autolog)
        mlflow.set_tag("task_type", "supervised")

        # Load data (datasets automatically logged by loader!)
        print("\n" + "-" * 60)
        print("Loading data...")
        X_train, X_test, y_train, y_test, X_val, y_val, loader = load_data(args)

        print(f"✓ Train: {len(X_train)} samples")
        print(f"✓ Test: {len(X_test)} samples")
        if X_val is not None:
            print(f"✓ Validation: {len(X_val)} samples")

        # Print data summary
        print(loader.summary())

        # Log data loader metadata (not handled by autolog)
        print("\n" + "-" * 60)
        print("Logging data loader metadata...")
        data_info = loader.get_data_info()
        for key, value in data_info.items():
            mlflow.set_tag(key, str(value))
        print("✓ Data loader metadata logged as tags")

        # Train model
        print("\n" + "-" * 60)
        model = train_model(X_train, y_train, args)

        # Evaluate model
        print("\n" + "-" * 60)
        metrics = evaluate_model(model, X_test, y_test, X_val, y_val)

        # Save metrics locally for DVC
        print("\n" + "-" * 60)
        print("Saving artifacts...")
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("✓ Metrics saved to metrics.json")

        # NOTE: Model is already logged by autolog!
        # No need to manually save or log model
        print("✓ Model automatically logged by autolog")
        print(f"✓ Model registered as '{args.model_name}' by autolog")

        print("\n" + "=" * 60)
        print(f"✓ TRAINING COMPLETE - Run ID: {run.info.run_id}")
        print("=" * 60)
        print("\nWhat was automatically logged:")
        print("  By Autolog:")
        print("    ✓ Model parameters (hyperparameters)")
        print("    ✓ Training metrics")
        print("    ✓ Model artifact")
        print("    ✓ Model signature")
        print("    ✓ Input examples")
        print("  By Data Loaders:")
        print("    ✓ Train dataset (context='training')")
        print("    ✓ Test dataset (context='testing')")
        if X_val is not None:
            print("    ✓ Validation dataset (context='validation')")
        print("    ✓ Dataset metadata (source, size, splits, etc.)")
        print("  Manually:")
        print("    ✓ Git metadata (commit SHA, branch, status)")
        print("    ✓ Task type and data loader info")

        return run.info.run_id


if __name__ == "__main__":
    main()
