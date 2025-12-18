# versioning

---

## Access Your Services

- lakeFS UI: http://localhost:8000
- MLflow UI: http://localhost:5001
- MinIO Console: http://localhost:9001

---

# MLOps Stack: Complete Versioning Solution

A production-ready MLOps implementation that provides full versioning for **code** (Git), **data** (lakeFS), and **models** (MLflow) with complete reproducibility.

## 🎯 Key Features

- **Data Versioning**: Git-like operations for data using lakeFS (branch, commit, merge, revert)
- **Model Tracking**: Full experiment tracking with MLflow including datasets, parameters, metrics, and models
- **Code Versioning**: Standard Git integration for code management
- **Reproducibility**: Every training run tracks exact versions of code, data, and parameters
- **Collaboration**: Multiple users can work on isolated data branches simultaneously
- **Production Ready**: Model registry with staging/production lifecycle management

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MLOps Stack                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │   Git    │    │  lakeFS  │    │  MLflow  │              │
│  │  (Code)  │───▶│  (Data)  │───▶│ (Models) │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                                              │
│  Versioned:       Versioned:      Versioned:                │
│  - Python code    - Datasets      - Parameters              │
│  - ML scripts     - Features      - Metrics                 │
│  - Configs        - Artifacts     - Models                  │
│                                    - Datasets (linked)       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Components

- **PostgreSQL**: Metadata storage for lakeFS and MLflow
- **MinIO**: S3-compatible object storage for data and model artifacts
- **lakeFS**: Data versioning with Git-like semantics
- **MLflow**: ML experiment tracking, model registry, and projects

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install Docker and Docker Compose
# Install Python 3.11+

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Start Services

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup (starts all services and creates repository)
./setup.sh
```

### 3. Set Environment Variables

```bash
export LAKEFS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export LAKEFS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
export LAKEFS_ENDPOINT=http://localhost:8000
export MLFLOW_TRACKING_URI=http://localhost:5000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin123
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
```

### 4. Initialize Git

```bash
git init
git add .
git commit -m "Initial commit"
```

### 5. Run First Pipeline

```bash
# Preprocess data and upload to lakeFS
python preprocess.py --lakefs_branch main

# Train model
python train.py --lakefs_branch main --max_depth 5 --n_estimators 100
```

## 🌐 Access UIs

- **lakeFS**: http://localhost:8000
  - Username: admin
  - Access Key: AKIAIOSFODNN7EXAMPLE
  - Secret Key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

- **MLflow**: http://localhost:5000

- **MinIO Console**: http://localhost:9001
  - Username: minioadmin
  - Password: minioadmin123

## 📚 Usage Examples

### Create Data Branch for Experiments

```bash
# Create branch
curl -X POST "http://localhost:8000/api/v1/repositories/ml-repo/branches" \
  -u "AKIAIOSFODNN7EXAMPLE:wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "experiment-1", "source": "main"}'

# Upload data to branch
python preprocess.py --lakefs_branch experiment-1

# Train on branch
python train.py --lakefs_branch experiment-1 --max_depth 10
```

### Run as MLflow Project

```bash
mlflow run . \
  -P lakefs_branch=main \
  -P max_depth=8 \
  -P n_estimators=120 \
  --env-manager=local
```

### Merge Successful Experiments

```bash
curl -X POST "http://localhost:8000/api/v1/repositories/ml-repo/refs/ml-repo/branches/experiment-1/merge/main" \
  -u "AKIAIOSFODNN7EXAMPLE:wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" \
  -H "Content-Type: application/json" \
  -d '{"message": "Merge successful experiment"}'
```

## 🔍 What Gets Versioned?

### Code (Git)
- Training scripts
- Preprocessing logic
- Model definitions
- Configuration files

### Data (lakeFS)
- Raw datasets
- Processed features
- Training/validation splits
- Data artifacts

### Models (MLflow)
- Model binaries
- Parameters
- Metrics
- Training metadata
- Dataset lineage (links to lakeFS commits)

## 🎓 Complete Testing Guide

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for comprehensive testing scenarios including:

1. Basic pipeline with data versioning
2. Data branching and experimentation
3. Model reproducibility
4. Merge data changes
5. MLflow Projects
6. Model registry and deployment
7. Disaster recovery (time travel)
8. Multi-user collaboration

## 📁 Project Structure

```
.
├── docker-compose.yml      # Service definitions
├── setup.sh               # Automated setup script
├── MLproject              # MLflow project definition
├── conda.yaml            # Environment specification
├── requirements.txt      # Python dependencies
├── preprocess.py         # Data preprocessing script
├── train.py             # Model training script
├── README.md            # This file
└── TESTING_GUIDE.md     # Detailed testing scenarios
```

## 🔄 Typical Workflow

```
1. Create data branch
   ↓
2. Modify/improve data
   ↓
3. Commit to lakeFS
   ↓
4. Train model (logs to MLflow)
   ↓
5. Compare with other experiments
   ↓
6. Merge successful data changes
   ↓
7. Register best model
   ↓
8. Promote to production
```

## 🛠️ Troubleshooting

### Services not starting

```bash
# Check Docker
docker-compose ps

# View logs
docker-compose logs lakefs
docker-compose logs mlflow

# Restart
docker-compose restart
```

### lakeFS connection issues

```bash
# Verify environment variables
env | grep LAKEFS

# Test connection
curl http://localhost:8000/_health
```

### MLflow tracking issues

```bash
# Verify tracking URI
echo $MLFLOW_TRACKING_URI

# Test connection
curl http://localhost:5000/health
```

## 🧹 Cleanup

```bash
# Stop services
docker-compose down

# Remove all data (WARNING: destructive)
docker-compose down -v
```

## 📈 Next Steps

After mastering the basics:

1. **Add CI/CD**: Automate testing and deployment with GitHub Actions
2. **Add DVC**: Local data caching for faster development
3. **Add Data Quality**: Implement Great Expectations for data validation
4. **Add Model Monitoring**: Track model performance in production
5. **Scale Up**: Deploy to Kubernetes for production use
6. **Add Security**: Implement proper authentication and authorization

## 🤝 Contributing

This is a reference implementation. Adapt it to your needs:

- Replace iris dataset with your own data
- Modify model architecture
- Add custom preprocessing
- Integrate with your existing tools

## 📄 License

This is an educational example for demonstrating MLOps best practices.

## 🙋 Support

For issues with:
- **lakeFS**: https://docs.lakefs.io
- **MLflow**: https://mlflow.org/docs
- **This implementation**: Create an issue or refer to TESTING_GUIDE.md
