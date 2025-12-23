# MLOps Template for Classical Machine Learning

A production-ready, generalized MLOps template for classical machine learning workflows (Supervised & Unsupervised).
This template integrates best practices for version control, experimentation tracking, and deployment.

## Core Components

1.  **Version Control**:

    - **Git**: Code versioning.
    - **DVC**: Data versioning with Git pre-hooks.
    - **Reproducibility**: Complete pipeline reproducibility through git commit hashes.

2.  **Experimentation Tracking**:

    - **MLflow**: Experiment tracking, model registry, dataset versioning.
    - **Advanced Aliasing**: Challenger/Champion model promotion.

3.  **Data Loading**:

    - Modular loaders for Tabular, Image, and Database sources.
    - Located in `src/data_loaders/`.

4.  **Training Scripts**:

    - Standalone, explicit training scripts derived from templates.
    - Located in `training/` (active script) and `templates/` (reference templates).

5.  **Deployment**:

    - **Ray Serve**: Online inference.
    - **Ray Data**: Batch prediction.
    - Located in `src/deployment/`.

6.  **CI/CD**:
    - GitLab CI/CD integration (`.gitlab-ci.yml`).
    - Pre-commit hooks for code quality and DVC.

## Directory Structure

```
├── .dvc/                   # DVC configuration
├── .github/                # GitHub Actions (if applicable)
├── .gitlab-ci.yml          # GitLab CI/CD pipeline
├── .pre-commit-config.yaml # Pre-commit hooks
├── dvc.yaml                # DVC pipeline stages
├── params.yaml             # DVC/MLflow parameters
├── training/               # Active training scripts
│   └── train.py            # Main training entry point
├── templates/              # Reference templates (Supervised, Unsupervised, Tuning)
├── src/
│   ├── data_loaders/       # Data loading modules
│   ├── deployment/         # Ray Serve/Data deployment
│   └── utils.py            # General utilities
├── scripts/                # CI/CD and helper scripts
└── data/                   # Data directory (DVC tracked)
```

## Setup

1.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Initialize Pre-commit Hooks**:

    ```bash
    pre-commit install
    ```

3.  **Configure DVC**:
    ```bash
    dvc init
    # Configure remote storage...
    ```

## Usage

### 1. Training (Supervised)

You can run training directly using the script:

```bash
python training/train.py \
    --data-path data/processed/dataset.csv \
    --target-column target \
    --experiment-name my-experiment
```

Or via DVC:

```bash
dvc repro train_supervised
```

### 2. Customizing Training

To change the model or logic, edit `training/train.py`.
You can also use other templates from `templates/` as a starting point:

- `templates/train_unsupervised.py`
- `templates/hyperparam_tuning.py`

### 3. Deployment

**Online Inference (Ray Serve)**:

```bash
python src/deployment/manager.py --type ray-serve-online
```

**Batch Inference (Ray Data)**:

```bash
python src/deployment/manager.py --type ray-batch --input data/new_data.csv --output data/predictions.csv
```

## CI/CD Pipeline

The `.gitlab-ci.yml` file defines the following stages:

1.  **Quality**: Code formatting, linting, security checks.
2.  **Validate**: Data validation.
3.  **Train**: Train challenger model.
4.  **Evaluate**: Compare challenger vs champion.
5.  **Deploy**: Build and push Docker images.
6.  **Release**: Tag and release.
7.  **Generate**: Generate new projects (Meta-MLOps).

## Project Generator (Meta-MLOps)

This repository includes a **Project Scaffolding Tool** that allows you to generate new, customized MLOps projects based on this template.

### How to Generate a New Project

**Option 1: Via GitLab CI/CD (Recommended)**

1.  Go to **Build** > **Pipelines**.
2.  Click **Run pipeline**.
3.  Set the following variables:
    - `PROJECT_NAME`: Name of the new project (e.g., `churn-prediction`).
    - `TASK_TYPE`: `supervised` or `unsupervised`.
    - `DATA_TYPE`: `tabular`, `image`, or `database`.
    - `DEPLOYMENT`: `all`, `ray-serve`, `ray-batch`, or `none`.
4.  Run the pipeline.
5.  When the `generate_repo` job completes, download the `artifacts.zip`.
6.  Extract it to start your new project!

**Option 2: Via CLI (Local)**
You can run the generator script locally:

```bash
python scripts/scaffold_project.py \
    --name my-new-project \
    --task-type supervised \
    --data-type tabular \
    --deployment all \
    --output-dir ../
```

### What it Generates

The tool creates a clean repository containing ONLY the components you need:

- **Tailored Training Script**: Pre-configured for your task type.
- **Specific Data Loader**: Only the loader you selected.
- **Deployment Configs**: Ray Serve/Batch files if requested.
- **CI/CD Pipeline**: customized `.gitlab-ci.yml`.
- **DVC & Params**: Configured `dvc.yaml` and `params.yaml`.

## Customization

- **Models**: Edit `training/train.py` directly.
- **Data**: Add new loaders in `src/data_loaders/`.
- **Pipeline**: Modify `dvc.yaml` to add new stages or parameters.
