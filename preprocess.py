"""
Data preprocessing script - uploads processed data to lakeFS
"""
import argparse
import os
import pandas as pd
from sklearn.datasets import load_iris
from lakefs import Repository, Branch
import lakefs
from io import BytesIO


def setup_lakefs():
    """Configure lakeFS client"""
    os.environ['LAKEFS_ACCESS_KEY_ID'] = os.getenv('LAKEFS_ACCESS_KEY_ID', 'AKIAIOSFODNN7EXAMPLE')
    os.environ['LAKEFS_SECRET_ACCESS_KEY'] = os.getenv('LAKEFS_SECRET_ACCESS_KEY', 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY')
    
    lakefs.config.host = os.getenv('LAKEFS_ENDPOINT', 'http://localhost:8000')
    return lakefs


def preprocess_data(args):
    """Load, preprocess, and upload data to lakeFS"""
    
    # Setup lakeFS
    setup_lakefs()
    
    # Load iris dataset
    print("Loading iris dataset...")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    
    # Simple preprocessing: normalize column names
    df.columns = df.columns.str.replace(' ', '_').str.replace('(cm)', '').str.strip()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset head:\n{df.head()}")
    
    # Upload to lakeFS
    repo = Repository('ml-repo')
    branch = Branch('ml-repo', args.lakefs_branch)
    
    # Convert DataFrame to CSV bytes
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    # Upload to lakeFS
    print(f"Uploading to lakeFS: ml-repo/{args.lakefs_branch}/data/iris_processed.csv")
    obj = branch.object('data/iris_processed.csv')
    obj.upload(csv_buffer, content_type='text/csv')
    
    # Commit the changes
    commit = branch.commit(
        message=f"Add processed iris dataset",
        metadata={'preprocessor': 'preprocess.py', 'rows': str(len(df))}
    )
    
    print(f"\n{'='*50}")
    print(f"Data uploaded successfully!")
    print(f"{'='*50}")
    print(f"lakeFS Commit ID: {commit.id}")
    print(f"Branch: {args.lakefs_branch}")
    print(f"Path: data/iris_processed.csv")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess data and upload to lakeFS')
    parser.add_argument('--lakefs_branch', type=str, default='main', help='lakeFS branch to upload to')
    
    args = parser.parse_args()
    preprocess_data(args)
