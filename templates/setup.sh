#!/bin/bash
# =============================================================================
# MLOps Project Setup Script
# =============================================================================
# This script initializes the environment, installs dependencies, and configures
# git hooks for DVC. Run this after cloning the repository.
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }

echo "=========================================="
echo "Initializing MLOps Project..."
echo "=========================================="

# 1. Check Git
if [ ! -d ".git" ]; then
    print_warning "Not a git repository. Initializing..."
    git init
    print_success "Git initialized"
fi

# 2. Setup Conda Environment
echo ""
echo "------------------------------------------"
echo "Setting up Conda Environment..."
echo "------------------------------------------"
if command -v conda &> /dev/null; then
    if conda env list | grep -q "mlops"; then
        print_warning "Conda environment 'mlops' already exists. Updating..."
        conda env update -f environment.yml --prune
    else
        echo "Creating 'mlops' environment..."
        conda create -f environment.yml -y
    fi
    print_success "Conda environment ready"
else
    print_error "Conda not found! Please install Miniconda/Anaconda first."
    exit 1
fi

# 3. Install Pre-commit Hooks
echo ""
echo "------------------------------------------"
echo "Installing Git Hooks..."
echo "------------------------------------------"
# Activate env temporarily to access pre-commit if installed there
# Or assume user has it globally or it's in the env we just created
# We'll try to use the one in the env
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mlops

if command -v pre-commit &> /dev/null; then
    pre-commit install
    print_success "Pre-commit hooks installed"
else
    print_warning "pre-commit not found in path. Installing..."
    pip install pre-commit
    pre-commit install
    print_success "Pre-commit hooks installed"
fi

# 4. Manual DVC Hooks (Post-Checkout/Merge)
# pre-commit handles 'pre-commit' and 'pre-push', but we need post-checkout/merge for DVC sync
echo "Configuring DVC auto-sync hooks..."

cat > .git/hooks/post-checkout << 'EOF'
#!/bin/bash
echo "🔄 Syncing DVC data after checkout..."
dvc checkout
EOF
chmod +x .git/hooks/post-checkout

cat > .git/hooks/post-merge << 'EOF'
#!/bin/bash
echo "🔄 Syncing DVC data after merge..."
dvc checkout
EOF
chmod +x .git/hooks/post-merge

print_success "DVC auto-sync hooks installed"

# 5. Initialize/Configure DVC
echo ""
echo "------------------------------------------"
echo "Configuring DVC..."
echo "------------------------------------------"
if [ ! -d ".dvc" ]; then
    dvc init
    print_success "DVC initialized"
else
    print_success "DVC already initialized"
fi

dvc config core.autostage true
print_success "DVC autostage enabled"

# 6. Verify Directory Structure
mkdir -p data/processed models
print_success "Directory structure verified"

# 7. Final Instructions
echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To start working:"
echo "  conda activate mlops"
echo ""
echo "To run the pipeline:"
echo "  dvc repro"
echo ""
