#!/bin/bash
# One-time setup script for reproducible ML pipeline

set -e

echo "=========================================="
echo "Reproducible ML Pipeline Setup"
echo "=========================================="

echo "Creating Conda environment from environment.yml..."
conda create -f environment.yml -y
echo "✓ Conda environment created from environment.yml"

# Initialize DVC (if not already)
if [ ! -d ".dvc" ]; then
    echo ""
    echo "Initializing DVC..."
    dvc init
    dvc config core.autostage true
    git add .dvc/ .dvcignore .gitignore
    git commit -m "Initialize DVC"
    echo "✓ DVC initialized"
else
    echo "✓ DVC already initialized"
fi

## Setup DVC remote
# Option 1: Local storage by default
if ! dvc remote list | grep -q "local-storage"; then
    echo ""
    echo "Configuring local DVC remote..."
    mkdir -p .dvc_storage
    dvc remote add -d local-storage .dvc_storage
    echo "✓ DVC remote local-storage configured"
fi

# Option 2: MinIO (S3-compatible, for shared storage) - That bucket must be already existing:
if ! dvc remote list | grep -q "remote-s3"; then
    echo ""
    echo "Configuring remote-s3 DVC remote..."
    dvc remote add -d remote-s3 s3://dvc-iris       # bucket name
    dvc remote modify remote-s3 endpointurl http://localhost:9000
    dvc remote modify remote-s3 access_key_id minioadmin
    dvc remote modify remote-s3 secret_access_key minioadmin123
    dvc remote modify remote-s3 use_ssl false
    echo "✓ DVC remote remote-s3 configured (MinIO)"
fi

# Track data with DVC
# if [ -f "data/dataset.csv" ] && [ ! -f "data/dataset.csv.dvc" ]; then
#     echo ""
#     echo "Adding data to DVC tracking..."
#     dvc add data/dataset.csv
#     echo "✓ Data tracked by DVC"
# elif [ -f "data/dataset.csv.dvc" ]; then
#     echo "✓ Data already tracked by DVC"
# else
#     echo "⚠️  No data/dataset.csv found. Create your dataset first."
# fi

# Configure DVC to auto-stage .dvc files
dvc config core.autostage true
echo "✓ DVC auto-stage enabled"

# Setup git hooks for automatic DVC sync
echo ""
echo "Setting up git hooks for automatic DVC sync..."

# Post-checkout hook
cat > .git/hooks/post-checkout << 'EOF'
#!/bin/bash
echo "🔄 Syncing DVC data after checkout..."
dvc checkout
if [ $? -eq 0 ]; then
    echo "✓ DVC data synced successfully"
else
    echo "⚠️  DVC checkout failed - you may need to run: dvc pull"
fi
EOF
chmod +x .git/hooks/post-checkout

# Post-merge hook
cat > .git/hooks/post-merge << 'EOF'
#!/bin/bash
echo "🔄 Syncing DVC data after merge..."
dvc checkout
if [ $? -eq 0 ]; then
    echo "✓ DVC data synced successfully"
else
    echo "⚠️  DVC checkout failed - you may need to run: dvc pull"
fi
EOF
chmod +x .git/hooks/post-merge

# Pre-push hook
cat > .git/hooks/pre-push << 'EOF'
#!/bin/bash
echo "🔄 Pushing DVC data to remote..."
dvc push
if [ $? -eq 0 ]; then
    echo "✓ DVC data pushed successfully"
else
    echo "⚠️  DVC push failed - data may not be synced to remote"
    echo "   Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
EOF
chmod +x .git/hooks/pre-push

echo "✓ Git hooks installed"

# # Update .gitignore
# echo ""
# echo "Updating .gitignore..."
# cat >> .gitignore << 'EOF'

# # DVC
# /data/dataset.csv
# .dvc_storage/

# # Models and outputs
# /models/
# /metrics.json

# # MLflow
# /mlruns/
# /mlartifacts/

# # Python
# __pycache__/
# *.pyc
# .pytest_cache/

# # IDE
# .vscode/
# .idea/
# EOF
# echo "✓ .gitignore updated"

# # Initialize git LFS for large files (optional)
# if command -v git-lfs &> /dev/null; then
#     echo ""
#     echo "Initializing Git LFS..."
#     git lfs install
#     echo "✓ Git LFS initialized"
# fi

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Commit DVC files:"
echo "   git add data/dataset.csv.dvc data/.gitignore .dvc/ .gitignore"
echo "   git commit -m 'Setup DVC tracking'"
echo ""
echo "2. Push data to DVC remote:"
echo "   dvc push"
echo ""
echo "3. Run training (choose one):"
echo "   "
echo "   Option A - MLflow Projects (recommended):"
echo "   mlflow run . -P n_estimators=200"
echo ""
echo "   Option B - DVC Pipeline:"
echo "   dvc repro"
echo ""
echo "   Option C - Direct Python:"
echo "   python -m src.train --n-estimators 200"
echo ""
echo "4. Test reproducibility:"
echo "   git checkout <old-commit>"
echo "   # Data auto-syncs via git hooks!"
echo "   dvc repro  # or: mlflow run ."
echo ""
