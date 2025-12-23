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
import warnings
import sys
import os

# Add project root to path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mlflow
import pandas as pd

from src.utils import get_git_metadata, validate_git_state

warnings.filterwarnings('ignore')

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))

# =============================================================================
# MLFLOW AUTOLOGGING (Uncomment the framework you're using)
# =============================================================================
# Generic autolog (works for most frameworks)
mlflow.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    disable=False,
    exclusive=False,  # Allow manual logging too
    silent=True
)

# =============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--target-column", type=str, required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)
    
    # Model hyperparameters
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    
    # MLflow parameters
    parser.add_argument("--experiment-name", type=str, default="my-experiment")
    parser.add_argument("--model-name", type=str, default="my-model")
    parser.add_argument("--strict-git", type=str, default="false")
    
    return parser.parse_args()


def load_data(args):
    """
    Load and split data using appropriate data loader.
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
        auto_log_mlflow=True  # Automatic dataset logging
    )
    
    # This automatically logs datasets to MLflow!
    X_train, X_test, y_train, y_test, X_val, y_val = loader.load_and_split()
    
    return X_train, X_test, y_train, y_test, X_val, y_val, loader


def train_model(X_train, y_train, args):
    """
    Train your model.
    """
    
    # Example: RandomForestClassifier
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    print("✓ Model trained")
    
    return model


def evaluate_model(model, X_test, y_test, X_val=None, y_val=None):
    """
    Evaluate model and return metrics.
    """
    
    print("Evaluating model...")
    
    # NOTE: Basic metrics are already logged by autolog!
    
    # Get basic metrics for display and saving (autolog already logged these)
    from sklearn.metrics import accuracy_score
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    
    metrics = {"test_accuracy": test_accuracy}
    
    print(f"\n✓ Evaluation complete")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    
    if X_val is not None and y_val is not None:
        val_accuracy = accuracy_score(y_val, model.predict(X_val))
        metrics["val_accuracy"] = val_accuracy
        print(f"  Val Accuracy: {val_accuracy:.4f}")
    
    return metrics


def main():
    """Main training pipeline."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("SUPERVISED LEARNING PIPELINE")
    print("="*60)
    
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
        print("\n" + "-"*60)
        print("Loading data...")
        X_train, X_test, y_train, y_test, X_val, y_val, loader = load_data(args)
        
        print(f"✓ Train: {len(X_train)} samples")
        print(f"✓ Test: {len(X_test)} samples")
        if X_val is not None:
            print(f"✓ Validation: {len(X_val)} samples")
        
        # Print data summary
        print(loader.summary())
        
        # Log data loader metadata (not handled by autolog)
        print("\n" + "-"*60)
        print("Logging data loader metadata...")
        data_info = loader.get_data_info()
        for key, value in data_info.items():
            mlflow.set_tag(key, str(value))
        print("✓ Data loader metadata logged as tags")
        
        # Train model
        print("\n" + "-"*60)
        model = train_model(X_train, y_train, args)
        
        # Evaluate model
        print("\n" + "-"*60)
        metrics = evaluate_model(model, X_test, y_test, X_val, y_val)
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        print("✓ Metrics logged to MLflow")
        
        # Save metrics locally for DVC
        print("\n" + "-"*60)
        print("Saving artifacts...")
        with open("metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        print("✓ Metrics saved to metrics.json")
        
        # NOTE: Model is already logged by autolog!
        # No need to manually save or log model
        print("✓ Model automatically logged by autolog")
        print(f"✓ Model registered as '{args.model_name}' by autolog")
        
        print("\n" + "="*60)
        print(f"✓ TRAINING COMPLETE - Run ID: {run.info.run_id}")
        print("="*60)
        
        return run.info.run_id


if __name__ == "__main__":
    main()
