# Training Templates 🎨

## Purpose

These are **starter templates** for data scientists. Pick the template for your task, customize it for your model, and start training!

**YOU ARE A DATA SCIENTIST, NOT AN MLOPS ENGINEER.**
Your job: Implement the model. The infrastructure is already set up.

---

## 📋 Available Templates

| Template | Use For | What to Implement |
|----------|---------|-------------------|
| `train_supervised.py` | Classification, Regression | Model architecture, evaluation metrics |
| `train_unsupervised.py` | Clustering, Dimensionality Reduction | Model type, clustering metrics |
| `train_hyperparam_tuning.py` | Hyperparameter Optimization | Search space, search method |
| `params.yaml.template` | All tasks | Your model's hyperparameters |
| `MLproject.template` | All tasks | Parameter definitions |
| `dvc.yaml.template` | All tasks | Pipeline stages |

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Choose Your Template

```bash
# For supervised learning (classification/regression)
cp templates/train_supervised.py train.py

# OR for unsupervised learning (clustering/etc)
cp templates/train_unsupervised.py train.py
```

### Step 2: Copy Config Templates

```bash
cp templates/params.yaml.template params.yaml
cp templates/MLproject.template MLproject
```

### Step 3: Open `train.py` and Follow Instructions

Look for `# TODO:` comments. You'll need to:

1. **Choose data loader** (delete the other 2)
   ```python
   # Keep one, delete two:
   from src.data_loaders import TabularDataLoader  # ← Keep this
   # from src.data_loaders import ImageDataLoader  # ← Delete
   # from src.data_loaders import DatabaseDataLoader  # ← Delete
   ```

2. **Implement your model**
   ```python
   def train_model(X_train, y_train, args):
       # TODO: Replace with your model
       import xgboost as xgb
       model = xgb.XGBClassifier(...)
       model.fit(X_train, y_train)
       return model
   ```

3. **Add hyperparameters**
   ```python
   def parse_args():
       # TODO: Add your hyperparameters
       parser.add_argument("--learning-rate", type=float, default=0.1)
       parser.add_argument("--n-estimators", type=int, default=100)
   ```

4. **Customize evaluation**
   ```python
   def evaluate_model(model, X_test, y_test):
       # TODO: Add your metrics
       metrics = {
           "accuracy": accuracy_score(y_test, y_pred),
           "auc": roc_auc_score(y_test, y_proba)
       }
       return metrics
   ```

### Step 4: Update `params.yaml` and `MLproject`

Add your hyperparameters in both files. See templates for examples.

### Step 5: Run!

```bash
mlflow run . -e supervised -P learning_rate=0.05
```

---

## 📖 What Each File Does

### `train_supervised.py` / `train_unsupervised.py`

**Your training script template.**

Contains:
- ✅ Data loading (3 examples - pick 1)
- ✅ MLflow setup (done for you)
- ✅ Git metadata tracking (done for you)
- ⚠️ Model training (YOU implement)
- ⚠️ Evaluation metrics (YOU implement)
- ⚠️ Hyperparameters (YOU add)

**What you do:**
1. Choose data loader
2. Implement model
3. Add hyperparameters
4. Customize metrics
5. Delete TODO comments

### `params.yaml.template`

**Hyperparameter configuration template.**

Contains:
- ✅ Data configuration section
- ⚠️ Model hyperparameters (YOU add)
- ✅ MLflow configuration

**What you do:**
1. Uncomment section for your task (supervised/unsupervised)
2. Add your model's hyperparameters
3. Delete unused sections

**Used by:** DVC pipeline (`dvc repro`)

### `MLproject.template`

**MLflow Projects configuration template.**

Contains:
- ✅ Environment setup (python_env.yaml)
- ⚠️ Entry points (YOU customize)
- ⚠️ Parameters with types (YOU add)
- ⚠️ Command to run (YOU update)

**What you do:**
1. Choose entry point (supervised/unsupervised)
2. Add your hyperparameters with types
3. Update command with your params
4. Delete unused entry points

**Used by:** MLflow Projects (`mlflow run .`)

---

## 🎯 Template Philosophy

### What Templates Provide ✅

**Infrastructure (You don't touch):**
- MLflow logging and tracking
- Git metadata extraction
- Data loader integration
- Model persistence
- Metrics logging
- Reproducibility tracking

### What You Implement ⚠️

**ML Logic (Your expertise):**
- Model selection and architecture
- Hyperparameter choices
- Feature engineering (if needed)
- Custom evaluation metrics
- Domain-specific logic

### What You Customize 🎨

**Configuration (Your decisions):**
- Which data loader to use
- Hyperparameter values
- Train/test split ratios
- Evaluation metrics
- Experiment names

---

## 📚 Examples by Model Type

### RandomForest (Supervised)

```python
# In train.py
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, args):
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
    model.fit(X_train, y_train)
    return model

def parse_args():
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
```

### XGBoost (Supervised)

```python
# In train.py
import xgboost as xgb

def train_model(X_train, y_train, args):
    model = xgb.XGBClassifier(
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        subsample=args.subsample
    )
    model.fit(X_train, y_train)
    return model
```

### KMeans (Unsupervised)

```python
# In train.py
from sklearn.cluster import KMeans

def train_model(X_train, args):
    model = KMeans(
        n_clusters=args.n_clusters,
        max_iter=args.max_iter,
        random_state=args.random_state
    )
    model.fit(X_train)
    return model
```

### Neural Network (Supervised)

```python
# In train.py
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def train_model(X_train, y_train, args):
    model = MyModel(
        input_dim=X_train.shape[1],
        hidden_dim=args.hidden_dim
    )
    # Training loop...
    return model
```

---

## 🔍 Template Anatomy

### What's Pre-Built (Don't Modify)

```python
def main():
    """Main training pipeline - MLOps infrastructure (don't modify)."""
    # This handles:
    # - Git metadata
    # - MLflow setup
    # - Logging
    # - Model persistence
    # - Reproducibility
```

### What You Implement

```python
def load_data(args):
    """INSTRUCTIONS: Choose loader, delete others."""
    # Pick one data loader
    # Delete the other two examples

def train_model(X_train, y_train, args):
    """INSTRUCTIONS: Implement your model."""
    # Replace placeholder with your model

def evaluate_model(model, X_test, y_test):
    """INSTRUCTIONS: Add your metrics."""
    # Add evaluation metrics for your task
```

---

## ⚠️ Common Mistakes

### ❌ Mistake 1: Not Deleting Unused Code

```python
# Don't leave all three loaders uncommented!
from src.data_loaders import TabularDataLoader  # Using this
from src.data_loaders import ImageDataLoader    # DELETE THIS
from src.data_loaders import DatabaseDataLoader # DELETE THIS
```

### ❌ Mistake 2: Forgetting to Add Hyperparameters

```python
# Must add to BOTH train.py and MLproject!

# In train.py
parser.add_argument("--learning-rate", type=float, default=0.1)

# In MLproject
learning_rate: {type: float, default: 0.1}
```

### ❌ Mistake 3: Modifying Infrastructure Code

```python
# DON'T modify this:
def main():
    """Main training pipeline - MLOps infrastructure (don't modify)."""
    # Leave this alone!

# DO modify this:
def train_model(X_train, y_train, args):
    # Implement your model here
```

---

## 📞 Need Help?

- **Data loader examples**: `../USAGE_EXAMPLES.md`
- **Complete guide**: `../DATA_SCIENTIST_GUIDE.md`
- **Template comments**: Read the `# TODO:` comments in each file
- **MLflow docs**: https://mlflow.org/docs/latest/projects.html

---

## 🎉 Success Criteria

You've successfully used the template when:

✅ You chose and kept ONE data loader
✅ You implemented YOUR model
✅ You added YOUR hyperparameters (in 3 places)
✅ You customized YOUR evaluation metrics
✅ You deleted the TODO comments
✅ Training runs via `mlflow run .`
✅ Results appear in MLflow UI

**Now you're doing ML, not infrastructure work! 🚀**
