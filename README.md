# Iris Classifier MLOps: Complete Reproducibility with Git + MLflow

A **reference implementation** demonstrating end-to-end reproducibility of code, data, and models using only **Git** and **MLflow**.

---

## 🎯 What This Project Demonstrates

1. **Full Reproducibility**: Any model can be traced back to exact code + data + hyperparameters
2. **Git as Source of Truth**: All code and data definitions versioned in Git
3. **MLflow for Tracking**: Experiments, datasets, models, and metadata tracked in MLflow
4. **Complete Traceability**: Every MLflow run links to a specific Git commit

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                         GIT REPOSITORY                       │
│  (Source of Truth for Code + Data Definitions)              │
│                                                              │
│  • Source code (train.py, data_loader.py, utils.py)        │
│  • Data file (data/iris.csv)                                │
│  • Environment (environment.yml)                            │
│  • Commit SHA = Unique version identifier                   │
└─────────────────────────────────────────────────────────────┘
                               │
                               │ Git metadata extracted
                               │ at runtime
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      TRAINING EXECUTION                      │
│  (src/train.py)                                             │
│                                                              │
│  1. Extract Git metadata (commit SHA, branch, etc.)         │
│  2. Load dataset and compute hash                           │
│  3. Train model with specified hyperparameters              │
│  4. Evaluate metrics                                        │
│  5. Log everything to MLflow                                │
└─────────────────────────────────────────────────────────────┘
                               │
                               │ All metadata + artifacts
                               │ logged to MLflow
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                         MLFLOW                               │
│  (Tracking & Registry for Experiments + Models)             │
│                                                              │
│  RUN METADATA:                                              │
│    • git.commit_sha                                         │
│    • git.branch                                             │
│    • git.repo_name                                          │
│    • dataset.name, dataset.version, dataset.hash           │
│    • hyperparameters (n_estimators, max_depth, etc.)       │
│    • metrics (accuracy, precision, recall, f1)             │
│                                                              │
│  ARTIFACTS:                                                  │
│    • Trained model (model/)                                 │
│    • Reproducibility manifest (CSV)                         │
│                                                              │
│  DATASETS (MLflow Datasets API):                            │
│    • iris-dataset with source + schema                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repository (or initialize new one)
git clone <your-repo-url>
cd iris-classifier-mlops

# Create conda environment
conda env create -f environment.yml
conda activate iris-mlops
```

### 2. Initialize Git (if new project)
```bash
git init
git add .
git commit -m "Initial commit: Iris classifier with Git + MLflow reproducibility"
```

### 3. Run Training
```bash
# Run with default parameters
python -m src.train

# Run with custom hyperparameters
python -m src.train \
    --n-estimators 200 \
    --max-depth 10 \
    --random-state 42
```

### 4. View Results in MLflow UI
```bash
mlflow ui
# Open http://localhost:5000 in browser
```

---

## 🔄 How Reproducibility Works

### The Reproducibility Chain
```
Git Commit SHA → Exact Code + Data → MLflow Run → Trained Model
     (1)              (2)                (3)           (4)

1. Git commit SHA uniquely identifies code version
2. Code defines data loading and transformation logic
3. MLflow run captures execution with all metadata
4. Model is artifact of that specific run
```

### Key Components

#### 1. **Git Metadata Extraction** (`src/utils.py`)
```python
git_metadata = get_git_metadata()
# Returns:
# {
#     "git.commit_sha": "a3b4c5d...",
#     "git.branch": "main",
#     "git.repo_name": "iris-classifier-mlops",
#     "git.is_dirty": "False"
# }
```

#### 2. **Dataset Versioning** (`src/data_loader.py`)

- Uses **MLflow Datasets API** to track dataset
- Computes **MD5 hash** of data file for versioning
- Logs dataset with name, source, and version to MLflow
```python
dataset = mlflow.data.from_pandas(df, source="data/iris.csv", name="iris-dataset")
mlflow.log_input(dataset, context="training")
```

#### 3. **Complete Run Tagging** (`src/train.py`)

Every MLflow run includes:
- Git commit SHA, branch, repo name
- Dataset name, version (hash), path
- All hyperparameters
- All metrics
- Reproducibility manifest (CSV artifact)

---

## 📊 Testing Reproducibility

### Test 1: Same Commit → Same Results
```bash
# Run training
python -m src.train --n-estimators 100 --max-depth 5
# Note the run ID and metrics

# Run again with same parameters
python -m src.train --n-estimators 100 --max-depth 5
# Results should be identical (same random_state)
```

### Test 2: Different Parameters → Different Results
```bash
# Run with different hyperparameters
python -m src.train --n-estimators 200 --max-depth 10
# Compare runs in MLflow UI
```

### Test 3: Modify Code → New Commit → Trackable Change
```bash
# Modify src/train.py (e.g., change default n_estimators)
git add src/train.py
git commit -m "Increase default n_estimators to 150"

# Run training
python -m src.train

# In MLflow UI, you'll see new git.commit_sha
# You can now compare this run to previous runs
```

### Test 4: Reproduce Old Run from Git History
```bash
# View MLflow UI and find an old run
# Note its git.commit_sha (e.g., a3b4c5d)

# Checkout that commit
git checkout a3b4c5d

# Re-run with same parameters (from run metadata)
python -m src.train --n-estimators 100 --max-depth 5

# Results should match the original run exactly
```

### Test 5: Modify Dataset → Detect Change
```bash
# Modify data/iris.csv (e.g., remove last 10 rows)
git add data/iris.csv
git commit -m "Reduce dataset size for testing"

# Run training
python -m src.train

# In MLflow UI, you'll see:
#   - Different git.commit_sha
#   - Different dataset.version (hash changed)
#   - Different dataset.rows
#   - Different metrics (due to smaller dataset)
```

---

## 🔍 Inspecting a Run in MLflow UI

For any run, you can see:

### Parameters
- `n_estimators`, `max_depth`, `random_state`
- `data.test_size`, `data.random_state`

### Metrics
- `train_accuracy`, `test_accuracy`
- `test_precision`, `test_recall`, `test_f1`

### Tags
- `git.commit_sha` ← **Use this to checkout exact code**
- `git.branch`
- `git.repo_name`
- `dataset.name`
- `dataset.version` ← **Hash of dataset**
- `dataset.hash` ← **Full MD5 hash**
- `dataset.path`

### Datasets
- Click "Datasets" tab to see `iris-dataset`
- View schema, source, and profile

### Artifacts
- `model/` - Trained model (can be loaded with `mlflow.sklearn.load_model()`)
- `reproducibility_manifest.csv` - Complete snapshot of run metadata

---

## 🎓 How Git and MLflow Complement Each Other

| Aspect | Git | MLflow |
|--------|-----|--------|
| **Code Versioning** | ✅ Source of truth | Records commit SHA |
| **Data Definition** | ✅ Stores data file or generation script | Tracks dataset hash + metadata |
| **Hyperparameters** | ❌ Not tracked | ✅ Logged as parameters |
| **Metrics** | ❌ Not tracked | ✅ Logged as metrics |
| **Models** | ❌ Binary files not practical | ✅ Model registry + artifacts |
| **Experiment Comparison** | ❌ Difficult | ✅ Built-in UI for comparison |
| **Reproducibility** | ✅ Enables exact code recreation | ✅ Enables exact run recreation |

**Together they provide**: Code + Data + Parameters + Metrics + Models = **Complete Reproducibility**

---

## 📁 Project Structure Explained

iris-classifier-mlops/
│
├── README.md                    # This file
├── environment.yml              # Conda environment specification
├── .gitignore                   # Git ignore rules
│
├── data/
│   └── iris.csv                # Dataset (versioned in Git)
│
├── src/
│   ├── init.py
│   ├── utils.py                # Git metadata extraction
│   ├── data_loader.py          # Dataset loading + MLflow logging
│   └── train.py                # Main training script
│
└── notebooks/
└── explore_runs.ipynb      # (Optional) Jupyter notebook for analysis

---

## 🛠️ Extending This Project

### Adding CI/CD Later

This project is designed to be **CI/CD-ready**. To add automation:

1. **GitHub Actions / GitLab CI**:
   - Trigger on push to main
   - Run `python -m src.train`
   - Upload MLflow artifacts to remote tracking server

2. **Remote MLflow Tracking**:
export MLFLOW_TRACKING_URI=https://your-mlflow-server.com
   python -m src.train
```

### Adding More Models

1. Create `src/train_model_b.py`
2. Use same Git + MLflow pattern
3. All runs are comparable in MLflow UI

### Adding More Datasets

1. Add new dataset in `data/`
2. Create new data loader class
3. Use `mlflow.log_input()` to track it

---

## ✅ Best Practices Demonstrated

1. **Always commit before training** (or use `--strict-git` flag)
2. **One experiment per repository** (keeps things organized)
3. **Programmatic Git metadata extraction** (no manual tagging)
4. **Dataset hashing for versioning** (detects data changes)
5. **Complete run manifests** (single artifact contains all metadata)
6. **Clear reproducibility instructions** (generated automatically)

---

## 🐛 Troubleshooting

### "Not a git repository" warning
- Initialize git: `git init && git add . && git commit -m "Initial commit"`

### MLflow UI not showing datasets
- Ensure you're using MLflow 2.9.2+
- Check that `mlflow.log_input()` was called

### Different results on re-run
- Check `random_state` parameter
- Verify dataset hasn't changed (check `dataset.hash` tag)
- Ensure same commit (`git.commit_sha`)

---

## 📚 Further Reading

- [MLflow Datasets Documentation](https://mlflow.org/docs/latest/ml/dataset/)
- [MLflow Tracking Documentation](https://mlflow.org/docs/latest/tracking.html)
- [Git Best Practices](https://git-scm.com/book/en/v2)

---

## 📄 License

MIT License - feel free to use this as a template for your ML projects.

---

## 🙋 Questions?

This project demonstrates **minimal but correct** reproducibility.
It's designed to be:
- Easy to understand
- Easy to run locally
- Easy to extend with CI/CD or orchestration tools later

**Key takeaway**: Git + MLflow provide complete reproducibility without additional complexity.
```

---

## 🧪 Step-by-Step Testing Guide

### Initial Setup
```bash
# 1. Create project directory
mkdir iris-classifier-mlops
cd iris-classifier-mlops

# 2. Create all files (copy code from above)
# ... create all files ...

# 3. Initialize Git
git init
git add .
git commit -m "Initial commit: Complete reproducibility system"

# 4. Create conda environment
conda env create -f environment.yml
conda activate iris-mlops
```

### Test Scenario 1: First Training Run
```bash
# Run training with default parameters
python -m src.train

# Expected output:
# - Git metadata printed
# - Dataset loaded and logged
# - Model trained
# - Metrics displayed
# - Reproducibility instructions printed

# Start MLflow UI
mlflow ui

# In browser (http://localhost:5000):
# - Navigate to "iris-classifier-mlops" experiment
# - Click on the run
# - Verify all tags are present (git.commit_sha, dataset.version, etc.)
# - Check "Datasets" tab shows iris-dataset
# - Download "reproducibility_manifest.csv" artifact
```

### Test Scenario 2: Compare Different Hyperparameters
```bash
# Run 1: Shallow forest
python -m src.train --n-estimators 50 --max-depth 3

# Run 2: Deep forest
python -m src.train --n-estimators 200 --max-depth 10

# In MLflow UI:
# - Select both runs
# - Click "Compare"
# - See parameter differences
# - See metric differences
# - Verify both have same git.commit_sha (same code)
```

### Test Scenario 3: Code Modification Tracking
```bash
# Modify code (e.g., change default n_estimators in train.py)
# Edit src/train.py: change default from 100 to 150

# Commit change
git add src/train.py
git commit -m "Change default n_estimators to 150"

# Run training
python -m src.train

# In MLflow UI:
# - Find this run
# - Compare git.commit_sha with previous runs
# - They will be different
# - Click commit SHA to see it in full
```

### Test Scenario 4: Full Reproducibility Test
```bash
# In MLflow UI, find your first run and note:
# - git.commit_sha (e.g., abc123...)
# - parameters used

# Checkout that exact commit
git checkout abc123

# Re-run with same parameters
python -m src.train --n-estimators 100 --max-depth 5

# In MLflow UI:
# - Compare this new run with original
# - Metrics should be identical
# - git.commit_sha should match
# - dataset.version should match
```

### Test Scenario 5: Dataset Change Detection
```bash
# Return to latest commit
git checkout main

# Modify dataset (add noise or remove rows)
python -c "import pandas as pd; df = pd.read_csv('data/iris.csv'); df = df.sample(frac=0.8); df.to_csv('data/iris.csv', index=False)"

# Commit change
git add data/iris.csv
git commit -m "Reduce dataset to 80% of original"

# Run training
python -m src.train

# In MLflow UI:
# - Compare with previous runs
# - dataset.version will be different (hash changed)
# - dataset.rows will be different
# - Metrics will likely be different
```

---

## 🎯 Key Learning Points

This implementation demonstrates:

1. **Git extracts code version** → Every run knows exact code used
2. **MLflow Datasets tracks data version** → Every run knows exact data used
3. **MLflow logs everything** → Complete audit trail
4. **Reproducibility is guaranteed** → Checkout commit + re-run = same result

**No CI/CD needed** - This works locally and forms a solid foundation for automation later.

**No DVC needed** - Dataset versioning via hashing + MLflow Datasets API.

**No complexity** - Just Git + MLflow + clear patterns.

