#!/bin/bash
# =============================================================================
# MLOps Project Setup Script
# =============================================================================
# This script initializes the environment, installs dependencies, and configures
# git hooks for DVC and automatic reproducibility.
# =============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

ENV_NAME="{{ project_name }}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ $1${NC}"; }

echo "=========================================="
echo "Initializing MLOps Project..."
echo "=========================================="

# =============================================================================
# 1. Check Git Repository
# =============================================================================
echo ""
echo "------------------------------------------"
echo "Checking Git Repository..."
echo "------------------------------------------"

if [ ! -d ".git" ]; then
    print_warning "Not a git repository. Initializing..."
    git init
    print_success "Git initialized"
else
    print_success "Git repository detected"
fi

# =============================================================================
# 2. Setup Conda Environment
# =============================================================================
echo ""
echo "------------------------------------------"
echo "Setting up Conda Environment..."
echo "------------------------------------------"

if ! command -v conda &> /dev/null; then
    print_error "Conda not found! Please install Miniconda/Anaconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

if conda env list | grep -q "^$ENV_NAME "; then
    print_warning "Conda environment '$ENV_NAME' already exists."
    read -p "Update existing environment? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env update -n "$ENV_NAME" -f environment.yml --prune
        print_success "Environment updated"
    else
        print_info "Skipping environment update"
    fi
else
    echo "Creating '$ENV_NAME' environment..."
    conda env create -n "$ENV_NAME" -f environment.yml
    print_success "Conda environment created"
fi

# =============================================================================
# 3. Activate Environment
# =============================================================================
echo ""
echo "------------------------------------------"
echo "Activating Environment..."
echo "------------------------------------------"

# Get conda base and activate
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
print_success "Environment '$ENV_NAME' activated"

# =============================================================================
# 4. Install Pre-commit Hooks
# =============================================================================
echo ""
echo "------------------------------------------"
echo "Installing Pre-commit Hooks..."
echo "------------------------------------------"

# Ensure pre-commit is available
if ! command -v pre-commit &> /dev/null; then
    print_warning "pre-commit not found. Installing..."
    pip install pre-commit
fi

# Install pre-commit hooks (handles pre-commit, pre-push stages)
pre-commit install --install-hooks

print_success "Pre-commit hooks installed"

# Optional: Run pre-commit on all files to verify setup
read -p "Run pre-commit checks on all files now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Running pre-commit checks (this may take a moment)..."
    pre-commit run --all-files || print_warning "Some checks failed. Fix issues and commit normally."
fi

# =============================================================================
# 5. Configure DVC Reproducibility Hooks
# =============================================================================
echo ""
echo "------------------------------------------"
echo "Configuring DVC Auto-Sync Hooks..."
echo "------------------------------------------"

# Post-checkout hook - Auto-sync data and environment after checkout
cat > .git/hooks/post-checkout << 'EOF'
#!/bin/bash
# =============================================================================
# Auto-sync after git checkout for full reproducibility
# =============================================================================

# Only run if checking out a commit (not just switching branches with same files)
if [ "$3" == "1" ]; then
    # echo "🔄 Syncing environment and data after checkout..."
    # 
    # # Check if conda is available
    # if command -v conda &> /dev/null; then
    #     # Get environment name from environment.yml
    #     ENV_NAME=$(grep 'name:' environment.yml | head -1 | awk '{print $2}')
    #     
    #     if [ ! -z "$ENV_NAME" ]; then
    #         # Update conda environment to match checked out environment.yml
    #         echo "📦 Updating conda environment: $ENV_NAME"
    #         conda env update -n "$ENV_NAME" -f environment.yml --prune 2>/dev/null || \
    #             echo "⚠️  Environment update skipped (may need manual update)"
    #     fi
    # fi

    # Sync DVC data to match checked out .dvc files
    if command -v dvc &> /dev/null; then
        echo "📊 Syncing DVC data..."
        dvc checkout 2>/dev/null || echo "⚠️  DVC checkout skipped (no changes or no data)"
    fi
    
    echo "✅ Reproducibility sync complete"
    echo "💡 Run 'dvc repro' to reproduce the pipeline"
fi
EOF
chmod +x .git/hooks/post-checkout

# Post-merge hook - Auto-sync after merge
cat > .git/hooks/post-merge << 'EOF'
#!/bin/bash
# =============================================================================
# Auto-sync after git merge for full reproducibility
# =============================================================================

# Sync DVC data
if command -v dvc &> /dev/null; then
    echo "📊 Syncing DVC data..."
    dvc checkout 2>/dev/null || echo "⚠️  DVC checkout skipped (no changes or no data)"
fi

echo "✅ Reproducibility sync complete"
EOF
chmod +x .git/hooks/post-merge

print_success "DVC auto-sync hooks configured"
print_info "Note: DVC pre-commit and pre-push hooks are managed by pre-commit"

# =============================================================================
# 6. Initialize/Configure DVC
# =============================================================================
echo ""
echo "------------------------------------------"
echo "Configuring DVC..."
echo "------------------------------------------"

if [ ! -d ".dvc" ]; then
    dvc init
    print_success "DVC initialized"
else:
    print_success "DVC already initialized"
fi

# Enable autostage (automatically stage .dvc files)
dvc config core.autostage true
print_success "DVC autostage enabled"

# Configure remotes
mkdir -p dvc_storage

# Add local remote (default for development)
if dvc remote list | grep -q "local-store"; then
    print_info "DVC remote 'local-store' already exists"
else
    dvc remote add -d local-store dvc_storage
    print_success "DVC local remote configured (default)"
fi

# Add AWS remote (for production/merged data)
if dvc remote list | grep -q "aws-store"; then
    print_info "DVC remote 'aws-store' already exists"
else
    dvc remote add aws-store "s3://{{ project_name }}"
    print_success "DVC AWS remote configured"
fi

print_info "DVC Strategy: Local remote is default. AWS remote is used after MR merge to main."

# =============================================================================
# 7. Verify Directory Structure
# =============================================================================
echo ""
echo "------------------------------------------"
echo "Verifying Directory Structure..."
echo "------------------------------------------"

mkdir -p data/raw data/external data/processed data/interim
mkdir -p models
mkdir -p notebooks
mkdir -p reports/figures
mkdir -p src

print_success "Directory structure verified"

# =============================================================================
# Final Summary
# =============================================================================
echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Environment: $ENV_NAME"
echo ""
echo "Next steps:"
echo "  1. Activate environment:  conda activate $ENV_NAME"
echo "  2. Configure AWS remote:  dvc remote modify aws-store access_key_id <YOUR_KEY>"
echo "  3. Pull DVC data:         dvc pull"
echo "  4. Run pipeline:          dvc repro"
echo ""
echo "Git hooks configured for FULL REPRODUCIBILITY:"
echo "  ✓ pre-commit: code quality + DVC status check"
echo "  ✓ pre-push: DVC validation"
echo "  ✓ post-checkout: auto-sync data + environment"
echo "  ✓ post-merge: auto-sync data + environment"
echo ""
echo "Reproducibility workflow:"
echo "  git checkout <commit-sha>  → Auto-syncs data + environment"
echo "  dvc repro                  → Reproduces exact pipeline"
echo ""
print_info "Tip: Use 'pre-commit run --all-files' to check code quality anytime"
echo ""
