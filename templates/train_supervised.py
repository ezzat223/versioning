"""
TEMPLATE: Supervised Learning Training Script

INSTRUCTIONS FOR DATA SCIENTISTS:
1. Choose your data loader (delete the other two examples)
2. Implement your model selection logic
3. Add your model's hyperparameters to params.yaml and MLproject
4. Implement your evaluation metrics
5. Delete these instructions and comments when ready

This template provides:
- Proper MLflow logging
- Git metadata tracking
- Data versioning with DVC
- Three data loader examples (pick one, delete others)
"""
import argparse
import json
import pickle
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from src.utils import get_git_metadata, validate_git_state

warnings.filterwarnings('ignore')


# =============================================================================
# MLFLOW AUTOLOGGING (Uncomment the framework you're using)
# =============================================================================
# Autologging automatically captures metrics, parameters, and models
# Comment out or delete the frameworks you're NOT using

# Generic autolog (works for most frameworks)
mlflow.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    disable=False,
    exclusive=False,  # Allow manual logging too
    silent=True
)

# Framework-specific autolog (uncomment if you need more control)
# import mlflow.sklearn
# mlflow.sklearn.autolog(
#     log_input_examples=True,
#     log_model_signatures=True,
#     log_models=True,
#     max_tuning_runs=5  # For hyperparameter tuning
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
#     log_models=True,
#     every_n_iter=1
# )

# import mlflow.pytorch
# mlflow.pytorch.autolog(
#     log_input_examples=True,
#     log_model_signatures=True,
#     log_models=True
# )

# import mlflow.keras
# mlflow.keras.autolog(
#     log_input_examples=True,
#     log_model_signatures=True,
#     log_models=True
# )

# import mlflow.fastai
# mlflow.fastai.autolog(
#     log_input_examples=True,
#     log_model_signatures=True,
#     log_models=True
# )

# import mlflow.statsmodels
# mlflow.statsmodels.autolog(
#     log_models=True
# )

# import mlflow.spark
# mlflow.spark.autolog(
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
        random_state=args.random_state
    )
    
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
    #     random_state=args.random_state
    # )
    # 
    # # Get splits (DataFrames with image paths)
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
    # # Setup your database connection
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
    #     random_state=args.random_state
    # )
    # 
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
    
    Example models:
    - RandomForestClassifier (sklearn)
    - XGBClassifier (xgboost)
    - LGBMClassifier (lightgbm)
    - PyTorch/TensorFlow models
    """
    
    # TODO: Replace this with your model!
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(
        n_estimators=100,  # TODO: Get from args
        max_depth=10,      # TODO: Get from args
        random_state=args.random_state
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    print("✓ Model trained")
    
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test, X_val=None, y_val=None):
    """
    Evaluate model and return metrics.
    
    INSTRUCTIONS:
    1. Add metrics relevant to your task (classification, regression, etc.)
    2. Consider adding custom metrics specific to your problem
    3. Return dictionary of metrics to log to MLflow
    """
    
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
