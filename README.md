# Meta-MLOps Project Generator

A production-ready **MLOps Project Generator** that scaffolds tailored machine learning repositories based on user specifications. This system acts as a "Cookiecutter on Steroids," allowing data scientists to generate fully configured MLOps projects with built-in best practices for version control, experimentation, and deployment.

## 🚀 Features

- **Dynamic Scaffolding**: Generates projects tailored to specific needs:
  - **Task Type**: Supervised (Classification/Regression) or Unsupervised (Clustering).
  - **Data Type**: Tabular, Image, or Database.
  - **Deployment**: Ray Serve (Online), Ray Data (Batch), or None.
- **Production-Ready Components**:
  - **Git + DVC**: Pre-configured for code and data versioning.
  - **MLflow**: Integrated experimentation tracking.
  - **CI/CD**: Auto-generated GitHub Actions pipelines for PRs and main.
  - **Code Quality**: Pre-commit hooks (Black, Flake8, Mypy, Bandit).
- **Automated Setup**: Includes a unified `setup.sh` script to bootstrap the environment in seconds.

---

## 🏗 Architecture

This repository operates as a **Generator** (Meta-MLOps). It does not contain the active model code itself but rather the _templates_ and _logic_ to create them.

```mermaid
graph LR
    A[User Inputs] --> B(Generator Script)
    B --> C{Templates}
    C --> D[Generated MLOps Repo]
    D --> E[CI/CD Pipeline]
    D --> F[Training & Deployment]
```

### Repository Structure

```
├── scripts/
│   └── scaffold_project.py   # The core generator logic (Jinja2-based)
├── templates/                # Jinja2 templates for the generated project
│   ├── dvc.yaml.template
│   ├── params.yaml.template
│   ├── github-actions.yml.template
│   ├── train_supervised.py
│   └── setup.sh
├── src/                      # Source code components (Data Loaders, Deployment)
│   ├── data_loaders/         # Modular data loaders to be copied
│   └── deployment/           # Deployment logic to be copied
└── requirements.txt          # Dependencies for the GENERATOR itself
```

---

## 🛠 Installation & Usage

### Prerequisites

- **Python 3.11+**
- **Conda** (Miniconda or Anaconda)
- **Git**

### Option 1: Local Generation (CLI)

1.  **Clone the Generator Repository**:

    ```bash
    git clone <generator-repo-url>
    cd meta-mlops-generator
    ```

2.  **Install Dependencies**:

    ```bash
    pip install jinja2
    ```

3.  **Run the Scaffolder**:
    Use the `scaffold_project.py` script to generate a new project.

    ```bash
    python scripts/scaffold_project.py \
      --name my-churn-model \
      --task-type supervised \
      --data-type tabular \
      --deployment ray-serve \
      --output-dir ../projects
    ```

    **Arguments**:

    - `--name`: Name of the new project (Required).
    - `--task-type`: `supervised` (default) or `unsupervised`.
    - `--data-type`: `tabular` (default), `image`, or `database`.
    - `--deployment`: `ray-serve`, `ray-batch`, `all`, or `none`.
    - `--output-dir`: Directory where the new project will be created.

4.  **Initialize the New Project**:
    Navigate to the created directory and run the setup script.
    ```bash
    cd ../projects/my-churn-model
    ./setup.sh
    ```
    _This will create the conda environment, install dependencies, and configure Git/DVC hooks._

### Option 2: CI/CD Generation (GitHub)

This repository includes a **GitHub Actions workflow** to generate and provision new project repositories automatically.

1. Configure repository secret:
   - Name: `GH_ADMIN_TOKEN`
   - Scopes: `repo`, `workflow`, and `admin:org` if creating under an org

2. Trigger the workflow:
   - Navigate to Actions → `Generate And Provision Project Repo` → `Run workflow`
   - Provide inputs:
     - `project_name` (string; letters, numbers, `.`, `_`, `-`)
     - `task_type` (`supervised` or `unsupervised`)
     - `data_type` (`tabular`, `image`, `database`)
     - `deployment` (`all`, `ray-serve`, `ray-batch`, `none`)
     - `owner` (optional; organization or username; defaults to meta repo owner)
     - `repo_private` (`true` or `false`; privacy of the new repo)

3. What the workflow does:
   - Generates the scaffold under `generated_projects/<project_name>/`
   - Creates a GitHub repository under `owner` if it does not exist
   - Pushes the scaffold to the repo’s `main` branch
   - Protects `main` with enforced admins and one required approving review

4. Start working:
   - `git clone https://github.com/<owner>/<project_name>.git`
   - `cd <project_name>`
   - `./setup.sh && conda activate <project_name>`
   - `dvc repro`

---

## 🧩 Customization

To modify the structure of _future_ generated projects, edit the files in the `templates/` directory:

- **`templates/dvc.yaml.template`**: Edit the DVC pipeline stages.
- **`templates/params.yaml.template`**: Add new hyperparameters or configuration sections.
- **`templates/github-actions.yml.template`**: Update the generated project’s CI/CD pipeline.
- **`src/`**: Add new data loaders or utility functions that should be copied to every new project.
