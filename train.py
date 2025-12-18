"""
Training script with lakeFS and MLflow integration
Versions: code (Git), data (lakeFS), models (MLflow)
"""
import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
from lakefs import Repository, Branch
import lakefs


def setup_lakefs():
    """Configure lakeFS client"""
    os.environ['LAKEFS_ACCESS_KEY_ID'] = os.getenv('LAKEFS_ACCESS_KEY_ID', 'AKIAIOSFODNN7EXAMPLE')
    os.environ['LAKEFS_SECRET_ACCESS_KEY'] = os.getenv('LAKEFS_SECRET_ACCESS_KEY', 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY')
    
    lakefs.config.host = os.getenv('LAKEFS_ENDPOINT', 'http://localhost:8000')
    return lakefs


def load_data_from_lakefs(repo_name, branch_name, file_path):
    """Load data from lakeFS repository"""
    print(f"Loading data from lakeFS: {repo_name}/{branch_name}/{file_path}")
    
    repo = Repository(repo_name)
    branch = Branch(repo_name, branch_name)
    
    # Get the file object
    obj = branch.object(file_path)
    
    # Read the data
    data = obj.reader().read()
    
    # Convert to DataFrame
    from io import StringIO
    df = pd.read_csv(StringIO(data.decode('utf-8')))
    
    # Get commit info for metadata
    commit = branch.head
    
    return df, {
        'lakefs_repo': repo_name,
        'lakefs_branch': branch_name,
        'lakefs_commit': commit.id,
        'lakefs_path': file_path
    }


def train_model(args):
    """Train model with full versioning"""
    
    # Setup lakeFS
    setup_lakefs()
    
    # Setup MLflow
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
    mlflow.set_experiment("iris-classification")
    
    with mlflow.start_run(run_name=f"train-{args.lakefs_branch}") as run:
        
        # Load data from lakeFS
        df, lakefs_metadata = load_data_from_lakefs(
            repo_name='ml-repo',
            branch_name=args.lakefs_branch,
            file_path='data/iris_processed.csv'
        )
        
        # Log lakeFS metadata
        mlflow.log_params(lakefs_metadata)
        
        # Log MLflow dataset
        dataset = mlflow.data.from_pandas(
            df,
            source=f"lakefs://ml-repo/{args.lakefs_branch}/data/iris_processed.csv",
            name="iris_dataset",
            targets="species"
        )
        mlflow.log_input(dataset, context="training")
        
        # Prepare data
        X = df.drop('species', axis=1)
        y = df['species']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Log parameters
        params = {
            'max_depth': args.max_depth,
            'n_estimators': args.n_estimators,
            'random_state': 42
        }
        mlflow.log_params(params)
        
        # Train model
        print(f"Training model with params: {params}")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted')
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="iris-classifier"
        )
        
        # Log code version (Git commit if available)
        try:
            import subprocess
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            mlflow.set_tag("git_commit", git_commit)
        except:
            print("Git repository not found, skipping git commit logging")
        
        print(f"\n{'='*50}")
        print(f"Training completed successfully!")
        print(f"{'='*50}")
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"Data Version (lakeFS commit): {lakefs_metadata['lakefs_commit']}")
        print(f"Metrics: {metrics}")
        print(f"{'='*50}\n")
        
        return run.info.run_id, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ML model with full versioning')
    parser.add_argument('--lakefs_branch', type=str, default='main', help='lakeFS branch to use')
    parser.add_argument('--max_depth', type=int, default=5, help='Max depth of trees')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees')
    
    args = parser.parse_args()
    train_model(args)
