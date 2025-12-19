"""
Main training script demonstrating full reproducibility with Git + MLflow.
"""
import argparse
import os
import warnings
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

from src.utils import get_git_metadata, validate_git_state, print_git_info
from src.data_loader import IrisDataLoader

# Suppress specific MLflow warnings
warnings.filterwarnings('ignore', category=UserWarning, module='mlflow.data.dataset_source_registry')
warnings.filterwarnings('ignore', category=UserWarning, module='mlflow.types.utils')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Iris classifier with full reproducibility")
    
    # Model hyperparameters
    parser.add_argument("--n-estimators", type=int, default=150, 
                        help="Number of trees in random forest")
    parser.add_argument("--max-depth", type=int, default=5, 
                        help="Maximum depth of trees")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Data parameters
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data for testing")
    parser.add_argument("--data-path", type=str, default="data/iris.csv",
                        help="Path to dataset")
    
    # MLflow parameters
    parser.add_argument("--experiment-name", type=str, default="iris-classifier-mlops-2",
                        help="MLflow experiment name")
    parser.add_argument("--model-name", type=str, default="iris-classifier",
                        help="Model name for MLflow registry")
    
    # Reproducibility validation
    parser.add_argument("--strict-git", action="store_true",
                        help="Fail if git working directory is dirty")
    
    return parser.parse_args()


def train_model(args):
    """
    Complete training workflow with full reproducibility tracking.
    """
    
    # ========================================
    # 1. EXTRACT GIT METADATA
    # ========================================
    print("\n" + "█"*60)
    print("STEP 1: EXTRACTING GIT METADATA")
    print("█"*60)
    
    git_metadata = get_git_metadata()
    print_git_info(git_metadata)
    
    # Validate git state (warn or fail on uncommitted changes)
    validate_git_state(git_metadata, strict=args.strict_git)
    
    
    # ========================================
    # 2. SETUP MLFLOW EXPERIMENT
    # ========================================
    print("\n" + "█"*60)
    print("STEP 2: SETTING UP MLFLOW EXPERIMENT")
    print("█"*60)
    
    # Set experiment (create if doesn't exist)
    mlflow.set_experiment(args.experiment_name)
    print(f"✓ Experiment: {args.experiment_name}")
    print(f"✓ Tracking URI: {mlflow.get_tracking_uri()}")
    
    
    # ========================================
    # 3. START MLFLOW RUN
    # ========================================
    with mlflow.start_run() as run:
        
        print(f"\n✓ MLflow Run ID: {run.info.run_id}")
        print(f"✓ Experiment ID: {run.info.experiment_id}")
        
        
        # ========================================
        # 4. LOG GIT METADATA TO MLFLOW
        # ========================================
        print("\n" + "█"*60)
        print("STEP 3: LOGGING GIT METADATA TO MLFLOW")
        print("█"*60)
        
        for key, value in git_metadata.items():
            mlflow.set_tag(key, value)
            print(f"✓ Tagged: {key} = {value}")
        
        
        # ========================================
        # 5. LOAD AND LOG DATASET
        # ========================================
        print("\n" + "█"*60)
        print("STEP 4: LOADING AND LOGGING DATASET")
        print("█"*60)
        
        data_loader = IrisDataLoader(
            data_path=args.data_path,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        # Load data splits
        X_train, X_test, y_train, y_test = data_loader.load_data()
        print(f"✓ Loaded data: {len(X_train)} train, {len(X_test)} test samples")
        
        # Log dataset to MLflow (with warning suppression)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_loader.log_to_mlflow()
        
        # Log data parameters
        mlflow.log_param("data.test_size", args.test_size)
        mlflow.log_param("data.random_state", args.random_state)
        
        # ========================================
        # 6. TRAIN MODEL
        # ========================================
        print("\n" + "█"*60)
        print("STEP 5: TRAINING MODEL")
        print("█"*60)
        
        # Log model hyperparameters
        params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "random_state": args.random_state,
        }
        mlflow.log_params(params)
        print(f"✓ Hyperparameters: {params}")
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state
        )
        model.fit(X_train, y_train)
        print("✓ Model trained")
        
        
        # ========================================
        # 7. EVALUATE MODEL
        # ========================================
        print("\n" + "█"*60)
        print("STEP 6: EVALUATING MODEL")
        print("█"*60)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "test_accuracy": accuracy_score(y_test, y_pred_test),
            "test_precision": precision_score(y_test, y_pred_test, average='weighted'),
            "test_recall": recall_score(y_test, y_pred_test, average='weighted'),
            "test_f1": f1_score(y_test, y_pred_test, average='weighted'),
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        print("✓ Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  - {metric_name}: {metric_value:.4f}")
        
        
        # ========================================
        # 8. LOG MODEL TO MLFLOW
        # ========================================
        print("\n" + "█"*60)
        print("STEP 7: LOGGING MODEL TO MLFLOW")
        print("█"*60)
        
        # Create model signature
        from mlflow.models import infer_signature
        signature = infer_signature(X_train, y_pred_train)
        
        # Log model with signature (suppress warnings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_info = mlflow.sklearn.log_model(
                model,
                name="model",
                signature=signature,
                registered_model_name=args.model_name
            )
        
        print(f"✓ Model logged and registered as '{args.model_name}'")
        print(f"✓ Model URI: {model_info.model_uri}")
        
        
        # ========================================
        # 9. GENERATE REPRODUCIBILITY MANIFEST
        # ========================================
        print("\n" + "█"*60)
        print("STEP 8: GENERATING REPRODUCIBILITY MANIFEST")
        print("█"*60)
        
        # Create comprehensive manifest
        manifest = {
            "mlflow_run_id": run.info.run_id,
            "mlflow_experiment_id": run.info.experiment_id,
            **git_metadata,
            **data_loader.get_metadata(),
            **params,
            **metrics,
        }
        
        # Save as artifact
        manifest_df = pd.DataFrame([manifest])
        manifest_path = "reproducibility_manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)
        mlflow.log_artifact(manifest_path)
        
        print("✓ Reproducibility manifest saved")
        print(f"\nMANIFEST SUMMARY:")
        print(f"  Run ID: {manifest['mlflow_run_id']}")
        print(f"  Git Commit: {manifest['git.commit_sha'][:8]}")
        print(f"  Dataset Version: {manifest['dataset.version']}")
        print(f"  Test Accuracy: {manifest['test_accuracy']:.4f}")
        
        # ========================================
        # 10. PRINT REPRODUCIBILITY INSTRUCTIONS
        # ========================================
        print("\n" + "█"*60)
        print("REPRODUCIBILITY INSTRUCTIONS")
        print("█"*60)
        print(f"""
To reproduce this exact run:

1. Checkout the git commit:
   git checkout {git_metadata['git.commit_sha']}

2. Activate the conda environment:
   conda activate iris-mlops

3. Re-run training with the same parameters:
   python -m src.train \\
       --n-estimators {args.n_estimators} \\
       --max-depth {args.max_depth} \\
       --random-state {args.random_state} \\
       --test-size {args.test_size}

4. View run in MLflow UI:
   mlflow ui --port 5001
   Open: http://localhost:5001
   
   Or view in remote server:
   Open: {mlflow.get_tracking_uri()}
""")
        
        print("█"*60 + "\n")
        
        return run.info.run_id


def main():
    """Main entry point."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("IRIS CLASSIFIER - REPRODUCIBLE ML TRAINING")
    print("="*60)
    
    try:
        run_id = train_model(args)
        
        print("\n" + "="*60)
        print(f"✓ TRAINING COMPLETE - Run ID: {run_id}")
        print("="*60)
        print("\nView results:")
        print(f"  Remote MLflow UI: {os.getenv('MLFLOW_TRACKING_URI')}")
        print("  Or local: mlflow ui --port 5001\n")
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ TRAINING FAILED")
        print("="*60)
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure Docker services are running: docker-compose ps")
        print("2. Check environment variables: source setup_env.sh")
        print("3. Test MLflow connection: curl http://localhost:5001/health")
        print("4. Test MinIO connection: curl http://localhost:9000/minio/health/live")
        print("="*60 + "\n")
        raise

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    main()
