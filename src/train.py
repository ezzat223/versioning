"""
Generic training script for supervised and unsupervised ML.
Automatically adapts based on whether target column is provided.
"""
import argparse
import json
import pickle
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,  # Supervised
    silhouette_score, davies_bouldin_score, calinski_harabasz_score  # Unsupervised
)

from src.data_loader import DataLoader
from src.utils import get_git_metadata, validate_git_state

warnings.filterwarnings('ignore', category=UserWarning)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ML model (supervised or unsupervised)")
    
    # Model selection
    parser.add_argument("--model-type", type=str, default="random_forest",
                       choices=["random_forest", "kmeans"],
                       help="Type of model: random_forest (supervised) or kmeans (unsupervised)")
    
    # Supervised model hyperparameters (RandomForest)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    
    # Unsupervised model hyperparameters (KMeans)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--n-init", type=int, default=10)
    
    # Data parameters
    parser.add_argument("--data-path", type=str, default="data/dataset.csv")
    parser.add_argument("--target-column", type=str, default=None,
                       help="Target column name (None for unsupervised learning)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.0)
    
    # Training parameters
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--experiment-name", type=str, default="ml-experiment")
    parser.add_argument("--model-name", type=str, default="ml-model")
    
    # Reproducibility
    parser.add_argument("--strict-git", type=str, default="false")
    
    return parser.parse_args()


def train_supervised(args, data_loader):
    """Train supervised learning model."""
    print("\n" + "-"*60)
    print("SUPERVISED LEARNING")
    print("-"*60)
    
    # Load data
    X_train, X_test, y_train, y_test, X_val, y_val = data_loader.load_and_split()
    print(f"✓ Train: {len(X_train)} samples")
    print(f"✓ Test: {len(X_test)} samples")
    if X_val is not None:
        print(f"✓ Validation: {len(X_val)} samples")
    
    # Log parameters
    params = {
        "model_type": args.model_type,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "random_state": args.random_state,
    }
    mlflow.log_params(params)
    
    # Train model
    print("\nTraining RandomForest...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state
    )
    model.fit(X_train, y_train)
    print("✓ Model trained")
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "test_precision": precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
        "test_recall": recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
        "test_f1": f1_score(y_test, y_pred_test, average='weighted', zero_division=0),
    }
    
    if X_val is not None:
        y_pred_val = model.predict(X_val)
        metrics["val_accuracy"] = accuracy_score(y_val, y_pred_val)
        metrics["val_f1"] = f1_score(y_val, y_pred_val, average='weighted', zero_division=0)
    
    mlflow.log_metrics(metrics)
    
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Log model signature
    from mlflow.models import infer_signature
    signature = infer_signature(X_train, y_pred_train)
    
    return model, metrics, signature


def train_unsupervised(args, data_loader):
    """Train unsupervised learning model."""
    print("\n" + "-"*60)
    print("UNSUPERVISED LEARNING")
    print("-"*60)
    
    # Load data (no targets)
    X_train, X_test, _, _, X_val, _ = data_loader.load_and_split()
    print(f"✓ Train: {len(X_train)} samples")
    print(f"✓ Test: {len(X_test)} samples")
    if X_val is not None:
        print(f"✓ Validation: {len(X_val)} samples")
    
    # Log parameters
    params = {
        "model_type": args.model_type,
        "n_clusters": args.n_clusters,
        "max_iter": args.max_iter,
        "n_init": args.n_init,
        "random_state": args.random_state,
    }
    mlflow.log_params(params)
    
    # Train model
    print(f"\nTraining KMeans (k={args.n_clusters})...")
    model = KMeans(
        n_clusters=args.n_clusters,
        max_iter=args.max_iter,
        n_init=args.n_init,
        random_state=args.random_state
    )
    model.fit(X_train)
    print("✓ Model trained")
    
    # Evaluate using clustering metrics
    print("\nEvaluating model...")
    train_labels = model.predict(X_train)
    test_labels = model.predict(X_test)
    
    metrics = {
        "train_silhouette": silhouette_score(X_train, train_labels),
        "train_davies_bouldin": davies_bouldin_score(X_train, train_labels),
        "train_calinski_harabasz": calinski_harabasz_score(X_train, train_labels),
        "test_silhouette": silhouette_score(X_test, test_labels),
        "test_davies_bouldin": davies_bouldin_score(X_test, test_labels),
        "test_calinski_harabasz": calinski_harabasz_score(X_test, test_labels),
        "train_inertia": model.inertia_,
    }
    
    if X_val is not None:
        val_labels = model.predict(X_val)
        metrics["val_silhouette"] = silhouette_score(X_val, val_labels)
    
    mlflow.log_metrics(metrics)
    
    print("\nClustering Metrics:")
    print(f"  Silhouette Score (higher is better): {metrics['test_silhouette']:.4f}")
    print(f"  Davies-Bouldin Index (lower is better): {metrics['test_davies_bouldin']:.4f}")
    print(f"  Calinski-Harabasz Score (higher is better): {metrics['test_calinski_harabasz']:.4f}")
    
    # Log model signature (cluster labels as output)
    from mlflow.models import infer_signature
    signature = infer_signature(X_train, train_labels)
    
    return model, metrics, signature


def train_model(args):
    """Main training workflow."""
    
    print("\n" + "="*60)
    print("REPRODUCIBLE ML TRAINING PIPELINE")
    print("="*60)
    
    # Extract and validate git metadata
    git_metadata = get_git_metadata()
    validate_git_state(git_metadata, strict=args.strict_git.lower() == "true")
    
    # Setup MLflow
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run() as run:
        print(f"\n✓ MLflow Run ID: {run.info.run_id}")
        
        # Log git metadata
        for key, value in git_metadata.items():
            mlflow.set_tag(key, value)
        
        # Determine task type
        is_supervised = args.target_column is not None
        task_type = "supervised" if is_supervised else "unsupervised"
        mlflow.set_tag("task_type", task_type)
        mlflow.set_tag("model_type", args.model_type)
        
        print(f"\nTask Type: {task_type.upper()}")
        print(f"Model Type: {args.model_type}")
        
        # Load data
        print("\n" + "-"*60)
        print("Loading data...")
        data_loader = DataLoader(
            data_path=args.data_path,
            target_column=args.target_column,
            test_size=args.test_size,
            validation_size=args.validation_size,
            random_state=args.random_state
        )
        
        # Log data info
        data_info = data_loader.get_data_info()
        for key, value in data_info.items():
            mlflow.log_param(key, value)
        
        # Print data summary
        print(data_loader.summary())
        
        # Train based on task type
        if is_supervised:
            model, metrics, signature = train_supervised(args, data_loader)
        else:
            model, metrics, signature = train_unsupervised(args, data_loader)
        
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
        metrics_path = Path("metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Metrics saved to {metrics_path}")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=args.model_name
        )
        print(f"✓ Model registered in MLflow as '{args.model_name}'")
        
        # Print reproducibility instructions
        print("\n" + "="*60)
        print("REPRODUCIBILITY")
        print("="*60)
        
        if is_supervised:
            repro_cmd = f"""mlflow run . \\
    -P model_type={args.model_type} \\
    -P target_column={args.target_column} \\
    -P n_estimators={args.n_estimators} \\
    -P max_depth={args.max_depth}"""
        else:
            repro_cmd = f"""mlflow run . \\
    -P model_type={args.model_type} \\
    -P n_clusters={args.n_clusters} \\
    -P max_iter={args.max_iter}"""
        
        print(f"""
To reproduce this run:

1. Using MLflow Projects:
{repro_cmd}

2. Using DVC:
   git checkout {git_metadata['git.commit_sha'][:8]}
   dvc repro

3. Using Git only:
   git checkout {git_metadata['git.commit_sha'][:8]}
   python -m src.train <args>

Data synced automatically via DVC git hooks!
""")
        
        return run.info.run_id


def main():
    """Entry point."""
    args = parse_args()
    
    try:
        run_id = train_model(args)
        print("\n" + "="*60)
        print(f"✓ TRAINING COMPLETE - Run ID: {run_id}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
