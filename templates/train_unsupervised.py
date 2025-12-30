"""
TEMPLATE: Unsupervised Learning Training Script

INSTRUCTIONS FOR DATA SCIENTISTS:
1. Choose your data loader (delete the other two examples)
2. Implement your model selection logic (clustering, dimensionality reduction, etc.)
3. Add your model's hyperparameters to params.yaml and MLproject
4. Implement your evaluation metrics
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

import mlflow

# Ensure project root is on PYTHONPATH when running as a script (e.g., dvc repro)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import get_git_metadata, validate_git_state

warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("http://127.0.0.1:5001")

# =============================================================================
# MLFLOW AUTOLOGGING
# =============================================================================
# Autologging automatically captures:
# - Model parameters (hyperparameters)
# - Model artifacts
# - Model signature
# - Input examples
#
# NOTE: For unsupervised learning, you'll need to manually log metrics
# as there are no standard metrics like accuracy for clustering/dimensionality reduction

mlflow.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    disable=False,
    exclusive=False,
    silent=True,
)
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Data parameters
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)

    # TODO: Add your model's hyperparameters
    # Example for clustering:
    # parser.add_argument("--n-clusters", type=int, default=3)
    # parser.add_argument("--max-iter", type=int, default=300)

    # Example for dimensionality reduction:
    # parser.add_argument("--n-components", type=int, default=2)
    # parser.add_argument("--learning-rate", type=float, default=200.0)

    # MLflow parameters
    parser.add_argument("--experiment-name", type=str, default="my-experiment")
    parser.add_argument("--model-name", type=str, default="my-model")
    parser.add_argument("--strict-git", type=str, default="false")

    return parser.parse_args()


def load_data(args):
    """
    Load and split data (no target for unsupervised).

    NOTE: Data loaders automatically log train/test/validation datasets to MLflow!

    Returns:
        X_train, X_test, X_val, loader
    """

    # ============================================================
    # OPTION 1: TABULAR DATA
    # ============================================================
    from src.data_loaders import TabularDataLoader

    loader = TabularDataLoader(
        data_path=args.data_path,
        target_column=None,  # No target for unsupervised!
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
        auto_log_mlflow=True,  # Automatic dataset logging
    )

    # This automatically logs datasets to MLflow!
    X_train, X_test, _, _, X_val, _ = loader.load_and_split()

    # ============================================================
    # OPTION 2: IMAGE DATA
    # ============================================================
    # from src.data_loaders import ImageDataLoader
    #
    # loader = ImageDataLoader(
    #     data_path=args.data_path,
    #     structure_type="directory",
    #     target_column=None,  # No target for unsupervised!
    #     image_size=(224, 224),
    #     test_size=args.test_size,
    #     validation_size=args.validation_size,
    #     random_state=args.random_state,
    #     auto_log_mlflow=True
    # )
    #
    # # This automatically logs datasets to MLflow!
    # X_train, X_test, _, _, X_val, _ = loader.load_and_split()
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
    #     target_column=None,  # No target for unsupervised!
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
    # X_train, X_test, _, _, X_val, _ = loader.load_and_split()

    return X_train, X_test, X_val, loader


def train_model(X_train, args):
    """
    Train unsupervised model.

    INSTRUCTIONS:
    1. Replace placeholder with your model (KMeans, DBSCAN, PCA, t-SNE, etc.)
    2. Use hyperparameters from args
    3. Return the trained model

    NOTE: Autolog will automatically log:
    - All hyperparameters
    - Model artifact
    - Model signature
    """

    # TODO: Replace with your model!
    # Example: Clustering
    from sklearn.cluster import KMeans

    model = KMeans(
        n_clusters=3, random_state=args.random_state, n_init=10  # TODO: Get from args.n_clusters
    )

    # Example: Dimensionality Reduction
    # from sklearn.decomposition import PCA
    # model = PCA(
    #     n_components=2,  # TODO: Get from args.n_components
    #     random_state=args.random_state
    # )

    # Example: Manifold Learning
    # from sklearn.manifold import TSNE
    # model = TSNE(
    #     n_components=2,
    #     learning_rate=200.0,  # TODO: Get from args
    #     random_state=args.random_state
    # )

    print("Training model...")
    model.fit(X_train)
    print("✓ Model trained")

    return model


def evaluate_model(model, X_train, X_test, X_val=None):
    """
    Evaluate unsupervised model.

    INSTRUCTIONS:
    Implement appropriate metrics for your unsupervised task:
    - Clustering: silhouette score, Davies-Bouldin index, inertia
    - Dimensionality reduction: explained variance, reconstruction error
    - Anomaly detection: contamination rate, precision/recall at threshold

    NOTE: Autolog does NOT capture unsupervised metrics automatically.
    You must manually log all metrics here.
    """

    print("Evaluating model...")

    metrics = {}

    # ============================================================
    # CLUSTERING METRICS
    # ============================================================
    if hasattr(model, "predict"):
        from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

        train_labels = model.predict(X_train)
        test_labels = model.predict(X_test)

        # Silhouette Score (higher is better, range: -1 to 1)
        metrics["train_silhouette_score"] = float(silhouette_score(X_train, train_labels))
        metrics["test_silhouette_score"] = float(silhouette_score(X_test, test_labels))

        # Davies-Bouldin Index (lower is better, range: 0 to inf)
        metrics["train_davies_bouldin_index"] = float(davies_bouldin_score(X_train, train_labels))
        metrics["test_davies_bouldin_index"] = float(davies_bouldin_score(X_test, test_labels))

        # Calinski-Harabasz Score (higher is better)
        metrics["train_calinski_harabasz_score"] = float(
            calinski_harabasz_score(X_train, train_labels)
        )
        metrics["test_calinski_harabasz_score"] = float(
            calinski_harabasz_score(X_test, test_labels)
        )

        # Inertia (for KMeans)
        if hasattr(model, "inertia_"):
            metrics["inertia"] = float(model.inertia_)

        if X_val is not None:
            val_labels = model.predict(X_val)
            metrics["val_silhouette_score"] = float(silhouette_score(X_val, val_labels))
            metrics["val_davies_bouldin_index"] = float(davies_bouldin_score(X_val, val_labels))

    # ============================================================
    # DIMENSIONALITY REDUCTION METRICS
    # ============================================================
    elif hasattr(model, "explained_variance_ratio_"):
        # PCA metrics
        metrics["explained_variance_ratio"] = float(model.explained_variance_ratio_.sum())
        metrics["n_components"] = int(model.n_components_)

        # Reconstruction error
        X_train_transformed = model.transform(X_train)
        X_train_reconstructed = model.inverse_transform(X_train_transformed)
        train_reconstruction_error = ((X_train - X_train_reconstructed) ** 2).mean()
        metrics["train_reconstruction_error"] = float(train_reconstruction_error)

        X_test_transformed = model.transform(X_test)
        X_test_reconstructed = model.inverse_transform(X_test_transformed)
        test_reconstruction_error = ((X_test - X_test_reconstructed) ** 2).mean()
        metrics["test_reconstruction_error"] = float(test_reconstruction_error)

    # ============================================================
    # ANOMALY DETECTION METRICS
    # ============================================================
    # elif hasattr(model, 'score_samples'):
    #     # For models like IsolationForest, LocalOutlierFactor
    #     train_scores = model.score_samples(X_train)
    #     test_scores = model.score_samples(X_test)
    #
    #     metrics["train_anomaly_score_mean"] = float(train_scores.mean())
    #     metrics["test_anomaly_score_mean"] = float(test_scores.mean())

    # Log all metrics to MLflow
    mlflow.log_metrics(metrics)

    print("\n✓ Evaluation complete")
    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    return metrics


def main():
    """Main training pipeline."""
    args = parse_args()

    print("\n" + "=" * 60)
    print("UNSUPERVISED LEARNING PIPELINE")
    print("=" * 60)

    # Git metadata
    git_metadata = get_git_metadata()
    validate_git_state(git_metadata, strict=args.strict_git.lower() == "true")

    # MLflow setup
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run() as run:
        print(f"\n✓ MLflow Run ID: {run.info.run_id}")
        print("✓ Autolog enabled")

        # Log git metadata
        print("\nLogging git metadata...")
        for key, value in git_metadata.items():
            mlflow.set_tag(key, value)
        print("✓ Git metadata logged")

        # Log task type
        mlflow.set_tag("task_type", "unsupervised")

        # Load data (datasets automatically logged by loader!)
        print("\n" + "-" * 60)
        print("Loading data...")
        X_train, X_test, X_val, loader = load_data(args)

        print(f"✓ Train: {len(X_train)} samples")
        print(f"✓ Test: {len(X_test)} samples")
        if X_val is not None:
            print(f"✓ Validation: {len(X_val)} samples")

        # Print data summary
        print(loader.summary())

        # Log data loader metadata
        print("\n" + "-" * 60)
        print("Logging data loader metadata...")
        data_info = loader.get_data_info()
        for key, value in data_info.items():
            mlflow.set_tag(key, str(value))
        print("✓ Data loader metadata logged as tags")

        # Train model
        print("\n" + "-" * 60)
        model = train_model(X_train, args)

        # Evaluate model
        print("\n" + "-" * 60)
        metrics = evaluate_model(model, X_train, X_test, X_val)

        # Save metrics locally for DVC
        print("\n" + "-" * 60)
        print("Saving artifacts...")
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print("✓ Metrics saved to metrics.json")

        # NOTE: Model is already logged by autolog!
        print("✓ Model automatically logged by autolog")
        print(f"✓ Model registered as '{args.model_name}' by autolog")

        print("\n" + "=" * 60)
        print(f"✓ TRAINING COMPLETE - Run ID: {run.info.run_id}")
        print("=" * 60)
        print("\nWhat was automatically logged:")
        print("  By Autolog:")
        print("    ✓ Model parameters (hyperparameters)")
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
        print("    ✓ Evaluation metrics (clustering/reduction metrics)")

        return run.info.run_id


if __name__ == "__main__":
    main()
