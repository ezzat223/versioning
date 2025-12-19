"""
Main training script demonstrating full reproducibility with Git + MLflow.
"""
import argparse
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

from src.utils import get_git_metadata, validate_git_state, print_git_info
from src.data_loader import IrisDataLoader

mlflow.set_tracking_uri("http://127.0.0.1:5001")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Iris classifier with full reproducibility")
    
    # Model hyperparameters
    parser.add_argument("--n-estimators", type=int, default=100, 
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
    parser.add_argument("--experiment-name", type=str, default="iris-classifier-mlops",
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
    
    
    # ========================================
    # 3. START MLFLOW RUN
    # ========================================
    with mlflow.start_run() as run:
        
        print(f"\n✓ MLflow Run ID: {run.info.run_id}")
        print(f"✓ Run URL: {mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
        
        
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
        
        # Log dataset to MLflow
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
        
        # Log model with signature
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=args.model_name
        )
        print(f"✓ Model logged and registered as '{args.model_name}'")
        
        
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
        print("\nMANIFEST CONTENTS:")
        for key, value in manifest.items():
            print(f"  {key:30s}: {value}")
        
        
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

4. Compare runs in MLflow UI:
   mlflow ui
   
   Then navigate to:
   {mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}
""")
        
        print("█"*60 + "\n")
        
        return run.info.run_id


def main():
    """Main entry point."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("IRIS CLASSIFIER - REPRODUCIBLE ML TRAINING")
    print("="*60)
    
    run_id = train_model(args)
    
    print("\n" + "="*60)
    print(f"✓ TRAINING COMPLETE - Run ID: {run_id}")
    print("="*60)
    print("\nView results in MLflow UI:")
    print("  mlflow ui")
    print("  Open: http://localhost:5000\n")


if __name__ == "__main__":
    main()
