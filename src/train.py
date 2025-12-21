"""
INSTRUCTIONS FOR DATA SCIENTISTS:
1. Choose your data loader (delete the other two examples)
2. Implement your model selection logic
3. Add your model's hyperparameters to params.yaml and MLproject
"""
import argparse
import json
import pickle
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from utils import get_git_metadata, validate_git_state
from data_loaders import TabularDataLoader

warnings.filterwarnings('ignore')


# Autologging configuration
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    max_tuning_runs=5,   # For hyperparameter tuning
    exclusive=False,     # Allow manual logging too
    silent=True,         # Suppress autologging errors
)


def parse_args():
    """Parse command line arguments (configured in MLproject)."""
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument("--data-path", type=str, default="data/processed/iris.csv")
    parser.add_argument("--target-column", type=str, default="target")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    
    # Model hyperparameters
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    
    # MLflow parameters
    parser.add_argument("--experiment-name", type=str, default="iris-classification")
    parser.add_argument("--model-name", type=str, default="iris-classifier")
    parser.add_argument("--strict-git", type=str, default="true")
    
    return parser.parse_args()


def load_data(args):
    """
    Load and split data using appropriate data loader.
    
    Returns:
        X_train, X_test, y_train, y_test, X_val, y_val, loader
    """
    
    loader = TabularDataLoader(
        data_path=args.data_path,
        target_column=args.target_column,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state
    )
    
    X_train, X_test, y_train, y_test, X_val, y_val = loader.load_and_split()
    
    return X_train, X_test, y_train, y_test, X_val, y_val, loader


def train_model(X_train, y_train, args):
    
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    print("✓ Model trained")
    
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test, X_val=None, y_val=None):
    
    # TODO: Replace with your evaluation metrics!
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    print("Evaluating model...")
    
    metrics = {
        "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
        "test_accuracy": accuracy_score(y_test, model.predict(X_test)),
        "test_f1": f1_score(y_test, model.predict(X_test), average='weighted'),
        "test_precision": precision_score(y_test, model.predict(X_test), average='weighted', zero_division=0),
        "test_recall": recall_score(y_test, model.predict(X_test), average='weighted', zero_division=0),
    }
    
    # Validation metrics if available
    if X_val is not None and y_val is not None:
        metrics["val_accuracy"] = accuracy_score(y_val, model.predict(X_val))
        metrics["val_f1"] = f1_score(y_val, model.predict(X_val), average='weighted')
    
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    return metrics


def main():
    """Main training pipeline - MLOps infrastructure (don't modify)."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("SUPERVISED LEARNING PIPELINE")
    print("="*60)
    
    # Git metadata
    git_metadata = get_git_metadata()
    validate_git_state(git_metadata, strict=args.strict_git.lower() == "true")
    
    # MLflow setup
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run() as run:
        print(f"\n✓ MLflow Run ID: {run.info.run_id}")
        
        # Log git metadata
        for key, value in git_metadata.items():
            mlflow.set_tag(key, value)
        
        mlflow.set_tag("task_type", "supervised")
        
        # Load data
        print("\n" + "-"*60)
        print("Loading data...")
        X_train, X_test, y_train, y_test, X_val, y_val, loader = load_data(args)
        
        print(f"✓ Train: {len(X_train)} samples")
        print(f"✓ Test: {len(X_test)} samples")
        if X_val is not None:
            print(f"✓ Validation: {len(X_val)} samples")
        
        # Print data summary
        print(loader.summary())
        
        # Log dataset to MLflow
        loader.log_to_mlflow(context="training")
        
        # Log data info
        for key, value in loader.get_data_info().items():
            mlflow.log_param(key, value)
        
        # TODO: Log your model hyperparameters
        # Example:
        # mlflow.log_params({
        #     "learning_rate": args.learning_rate,
        #     "n_estimators": args.n_estimators,
        # })
        
        # Train model
        print("\n" + "-"*60)
        model = train_model(X_train, y_train, args)
        
        # Evaluate model
        print("\n" + "-"*60)
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test, X_val, y_val)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Save model locally
        print("\n" + "-"*60)
        print("Saving model...")
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Model saved to {model_path}")
        
        # Save metrics
        with open("metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Log model to MLflow
        from mlflow.models import infer_signature
        signature = infer_signature(X_train, model.predict(X_train))
        
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=args.model_name
        )
        print(f"✓ Model registered as '{args.model_name}'")
        
        print("\n" + "="*60)
        print(f"✓ TRAINING COMPLETE - Run ID: {run.info.run_id}")
        print("="*60)
        
        return run.info.run_id


if __name__ == "__main__":
    main()
