# Steps

---

## In this template's repo

### PART 1

#### 1. Configure GitLab CI/CD Variables

**Go to: Settings → CI/CD → Variables and add:**
```bash
# MLflow Configuration
MLFLOW_TRACKING_URI: http://YOUR_PUBLIC_IP:5001  # Or use ngrok/cloudflare tunnel
DOCKER_HUB_USERNAME: your_docker_username
DOCKER_HUB_TOKEN: your_docker_token  # Type: Masked

# AWS/MinIO Configuration (for DVC remote)
AWS_ACCESS_KEY_ID: minioadmin
AWS_SECRET_ACCESS_KEY: minioadmin123  # Type: Masked
AWS_DEFAULT_REGION: us-east-1

## Important: If MLflow is on localhost, you need to expose it:
```

#### 2. Protect Main Branch
**Go to: Settings → Repository → Protected Branches**
Branch: main
✅ Allowed to merge: Maintainers
✅ Allowed to push: No one

### PART 2: MLOps Engineer - Generate Project

#### 1. Trigger Project Generation

Go to: CI/CD → Pipelines → Run Pipeline
**Set variables:**
- PROJECT_NAME: <project-or-repo-name>
- TASK_TYPE: supervised
- DATA_TYPE: tabular
- DEPLOYMENT: batch

Click Run Pipeline

#### 2. Download Generated Project
Once pipeline succeeds:

Go to CI/CD → Pipelines → [Your pipeline] → Job: generate_repo
Download artifact: generated_projects/<project-or-repo-name>.tar.gz
>Open it in vscode

#### 3. GitLab repo

- Create a GitLab repo with same name of the project
- Follow up with the steps of `Push an existing folder`

---

## In the generated project

### 1. Configure CI/CD Variables (iris-classifier repo)

Go to: Settings → CI/CD → Variables (in iris-classifier repo)
```bash
MLFLOW_TRACKING_URI: http://YOUR_PUBLIC_IP:5001
EXPERIMENT_NAME: iris-classifier-exp
MODEL_NAME: iris-classifier-model
PRIMARY_METRIC: test_accuracy
DATA_PATH: data/processed/iris.csv
TARGET_COLUMN: species

# Docker Hub credentials
DOCKER_HUB_USERNAME: your_username
DOCKER_HUB_TOKEN: your_token  # Masked

# AWS/MinIO for DVC
AWS_ACCESS_KEY_ID: minioadmin
AWS_SECRET_ACCESS_KEY: minioadmin123  # Masked
```

### 2. Protect Main Branch (iris-classifier repo)

Settings → Repository → Protected Branches

Branch: main
✅ Allowed to merge: Maintainers
✅ Allowed to push: No one

---

## Data Scientists

### Setup Local Environment

```bash
./setup.sh  # Answer prompts

# If faced any problems while creating the conda environment, try:
conda clean --all -y

# Activate environment
conda deactivate # till you are out of all even base
conda activate iris-classifier
```

### Data and DVC

```bash
# Track with DVC
dvc add data/processed/iris.csv
dvc add data/processed/iris_test_features.csv
dvc add data/processed/iris_test_full.csv

# Commit
git add data/processed/.gitignore data/processed/*.dvc scripts/prepare_iris_data.py
git commit -m "feat: Add Iris dataset preparation"
```

### Data Validation (sripts/validate_data.py)

To test it after customization
```bash
python scripts/validate_data.py --data data/processed/iris.csv
```
