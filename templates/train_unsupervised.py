"""
TEMPLATE: Unsupervised Learning Training Script

INSTRUCTIONS FOR DATA SCIENTISTS:
1. Choose your data loader (delete the other two examples)
2. Implement your model selection logic (clustering, dimensionality reduction, etc.)
3. Add your model's hyperparameters to params.yaml and MLproject
4. Implement your evaluation metrics (silhouette score, inertia, etc.)
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

from src.utils import get_git_metadata, validate_git_state

warnings.filterwarnings('ignore')

mlflow.set_tracking_uri("http://localhost:5001")

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
#     log_models=True
# )

# See train_supervised.py for more framework examples
# =============================================================================


def parse_args():
    """Parse command line arguments (configured in MLproject)."""
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)
    
    # TODO: Add your model's hyperparameters here
    # Example for clustering:
    # parser.add_argument("--n-clusters", type=int, default=3)
    # parser.add_argument("--max-iter", type=int, default=300)
    
    # Example for dimensionality reduction:
    # parser.add_argument("--n-components", type=int, default=2)
    
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
    4. Note: target_column=None for unsupervised learning
    
    Returns:
        X_train, X_test, X_val, loader
    """
    
    # ============================================================
    # OPTION 1: TABULAR DATA (CSV, Parquet, Excel)
    # ============================================================
    from src.data_loaders import TabularDataLoader
    
    loader = TabularDataLoader(
        data_path=args.data_path,
        target_column=None,  # No target for unsupervised!
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state
    )
    
    X_train, X_test, _, _, X_val, _ = loader.load_and_split()
    
    # ============================================================
    # OPTION 2: IMAGE DATA
    # ============================================================
    # from src.data_loaders import ImageDataLoader
    # 
    # loader = ImageDataLoader(
    #     data_path=args.data_path,
    #     structure_type="directory",  # Flat directory for unsupervised
    #     target_column=None,  # No target for unsupervised!
    #     image_size=(224, 224),
    #     test_size=args.test_size,
    #     validation_size=args.validation_size,
    #     random_state=args.random_state
    # )
    # 
    # X_train, X_test, _, _, X_val, _ = loader.load_and_split()
    # 
    # # TODO: Load actual images or extract features
    # # images_train = loader.load_images(X_train)
    # # Or extract features using pretrained CNN
    
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
    #     target_column=None,  # No target for unsupervised!
    #     database_type="postgresql",
    #     cache_data=True,
    #     test_size=args.test_size,
    #     validation_size=args.validation_size,
    #     random_state=args.random_state
    # )
    # 
    # X_train, X_test, _, _, X_val, _ = loader.load_and_split()
    
    return X_train, X_test, X_val, loader


def train_model(X_train, args):
    """
    Train your unsupervised model.
    
    INSTRUCTIONS:
    1. Replace this placeholder with your actual model
    2. Common unsupervised models:
       - KMeans, DBSCAN, Hierarchical (clustering)
       - PCA, t-SNE, UMAP (dimensionality reduction)
       - Autoencoders (deep learning)
       - Isolation Forest (anomaly detection)
    3. Use hyperparameters from args
    
    Returns:
        trained model
    """
    
    # TODO: Replace with your model!
    # Example: Clustering
    from sklearn.cluster import KMeans
    
    model = KMeans(
        n_clusters=3,  # TODO: Get from args
        max_iter=300,  # TODO: Get from args
        random_state=args.random_state
    )
    
    # Example: Dimensionality Reduction
    # from sklearn.decomposition import PCA
    # model = PCA(n_components=2)
    
    # Example: Anomaly Detection
    # from sklearn.ensemble import IsolationForest
    # model = IsolationForest(contamination=0.1, random_state=args.random_state)
    
    print("Training model...")
    model.fit(X_train)
    print("✓ Model trained")
    
    return model


def evaluate_model(model, X_train, X_test, X_val=None):
    """
    Evaluate unsupervised model.
    
    INSTRUCTIONS:
    1. Choose metrics appropriate for your task:
       - Clustering: silhouette_score, davies_bouldin_score, calinski_harabasz_score
       - Dimensionality reduction: explained_variance_ratio
       - Anomaly detection: contamination rate, scores distribution
    2. Return dictionary of metrics
    """
    
    print("Evaluating model...")
    
    # TODO: Replace with your evaluation metrics!
    
    # Example: Clustering metrics
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    
    train_labels = model.predict(X_train)
    test_labels = model.predict(X_test)
    
    metrics = {
        "train_silhouette": silhouette_score(X_train, train_labels),
        "train_davies_bouldin": davies_bouldin_score(X_train, train_labels),
        "train_calinski_harabasz": calinski_harabasz_score(X_train, train_labels),
        "test_silhouette": silhouette_score(X_test, test_labels),
        "test_davies_bouldin": davies_bouldin_score(X_test, test_labels),
        "test_calinski_harabasz": calinski_harabasz_score(X_test, test_labels),
    }
    
    # Add model-specific metrics
    if hasattr(model, 'inertia_'):
        metrics["train_inertia"] = model.inertia_
    
    # Validation metrics
    if X_val is not None:
        val_labels = model.predict(X_val)
        metrics["val_silhouette"] = silhouette_score(X_val, val_labels)
    
    # Example: Dimensionality Reduction metrics
    # if hasattr(model, 'explained_variance_ratio_'):
    #     metrics["explained_variance"] = sum(model.explained_variance_ratio_)
    
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    return metrics


def main():
    """Main training pipeline - MLOps infrastructure (don't modify)."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("UNSUPERVISED LEARNING PIPELINE")
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
        
        mlflow.set_tag("task_type", "unsupervised")
        
        # Load data
        print("\n" + "-"*60)
        print("Loading data...")
        X_train, X_test, X_val, loader = load_data(args)
        
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
        # mlflow.log_params({
        #     "n_clusters": args.n_clusters,
        #     "max_iter": args.max_iter,
        # })
        
        # Train model
        print("\n" + "-"*60)
        model = train_model(X_train, args)
        
        # Evaluate model
        print("\n" + "-"*60)
        metrics = evaluate_model(model, X_train, X_test, X_val)
        
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
            name="model",
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
