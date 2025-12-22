#!/bin/bash
# Setup script for CI/CD pipeline
# Run this once to configure your project for CI/CD

set -e

echo "=========================================="
echo "CI/CD Pipeline Setup"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    print_error "Not a git repository. Please run 'git init' first."
    exit 1
fi

print_success "Git repository detected"

# Install pre-commit hooks
echo ""
echo "Installing pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    print_success "Pre-commit hooks installed"
else
    print_warning "pre-commit not found. Install with: pip install pre-commit"
fi

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p scripts
mkdir -p data/processed
mkdir -p models
print_success "Directories created"

# Check for required files
echo ""
echo "Checking required files..."

required_files=(
    ".gitlab-ci.yml"
    "scripts/compare_models.py"
    "scripts/promote_model.py"
    "scripts/data_validation.py"
    "scripts/generate_cml_report.py"
    "Dockerfile"
    "requirements.txt"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        print_success "$file exists"
    else
        print_error "$file missing"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    print_error "Missing required files. Please ensure all CI/CD files are in place."
    exit 1
fi

# Update .gitlab-ci.yml with user's Docker Hub username
echo ""
echo "=========================================="
echo "Docker Hub Configuration"
echo "=========================================="
read -p "Enter your Docker Hub username: " docker_username

if [ ! -z "$docker_username" ]; then
    # Update .gitlab-ci.yml
    sed -i.bak "s/your-dockerhub-username/${docker_username}/g" .gitlab-ci.yml
    print_success "Updated Docker Hub username in .gitlab-ci.yml"
    rm .gitlab-ci.yml.bak 2>/dev/null || true
else
    print_warning "Skipped Docker Hub configuration"
fi

# GitLab CI/CD Variables reminder
echo ""
echo "=========================================="
echo "GitLab CI/CD Variables Setup"
echo "=========================================="
echo ""
echo "Please configure these variables in GitLab:"
echo "Settings → CI/CD → Variables"
echo ""
echo "Required Variables:"
echo "  DOCKER_HUB_USERNAME   - Your Docker Hub username"
echo "  DOCKER_HUB_TOKEN      - Docker Hub access token (masked)"
echo ""
echo "Optional Variables:"
echo "  MLFLOW_TRACKING_URI   - MLflow server URL"
echo "  DVC_REMOTE_URL        - DVC remote storage URL"
echo ""

# Test script permissions
echo "=========================================="
echo "Setting script permissions..."
chmod +x scripts/*.py 2>/dev/null || true
chmod +x scripts/*.sh 2>/dev/null || true
print_success "Script permissions set"

# Create example .env file
echo ""
echo "Creating .env.example..."
cat > .env.example << 'EOF'
# MLflow Configuration
MLFLOW_TRACKING_URI=http://127.0.0.1:5001
EXPERIMENT_NAME=iris-classification-ci
MODEL_NAME=iris-classifier-ci

# Docker Configuration
DOCKER_IMAGE_NAME=your-username/iris-classifier
MODEL_VERSION=latest

# DVC Configuration
DVC_REMOTE_URL=s3://your-bucket

# API Configuration
PORT=8000
EOF
print_success ".env.example created"

# Git configuration
echo ""
echo "=========================================="
echo "Git Configuration"
echo "=========================================="

# Add files to git
git add .gitlab-ci.yml scripts/ Dockerfile requirements.txt .dockerignore .pre-commit-config.yaml pyproject.toml 2>/dev/null || true
print_success "Files staged for commit"

echo ""
echo "Suggested git commands:"
echo "  git commit -m 'Setup CI/CD pipeline'"
echo "  git push origin main"

# Summary
echo ""
echo "=========================================="
echo "✅ CI/CD Setup Complete!"
echo "=========================================="
echo ""
echo "Next Steps:"
echo ""
echo "1. Configure GitLab CI/CD Variables:"
echo "   → Settings → CI/CD → Variables"
echo "   → Add: DOCKER_HUB_USERNAME, DOCKER_HUB_TOKEN"
echo ""
echo "2. Commit and push changes:"
echo "   git add ."
echo "   git commit -m 'Setup CI/CD pipeline'"
echo "   git push origin main"
echo ""
echo "3. Monitor pipeline:"
echo "   → CI/CD → Pipelines"
echo ""
echo "4. Review documentation:"
echo "   → docs/CI_CD_GUIDE.md"
echo ""
echo "=========================================="
echo ""

# Offer to open documentation
read -p "Open CI/CD guide in browser? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "docs/CI_CD_GUIDE.md" ]; then
        # Try different markdown viewers
        if command -v mdcat &> /dev/null; then
            mdcat docs/CI_CD_GUIDE.md
        elif command -v glow &> /dev/null; then
            glow docs/CI_CD_GUIDE.md
        else
            cat docs/CI_CD_GUIDE.md
        fi
    fi
fi

echo ""
print_success "Setup complete! Happy MLOps! 🚀"
