#!/usr/bin/env python3
"""
MLOps Project Scaffolder - IMPROVED
Generates production-ready MLOps projects with proper error handling.
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scaffold a new MLOps project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic supervised learning project
  python scaffold_project.py --name my-churn-model --task-type supervised
  
  # Image classification with Ray Serve deployment
  python scaffold_project.py --name image-classifier --task-type supervised \\
      --data-type image --deployment ray-serve
  
  # Clustering project with no deployment
  python scaffold_project.py --name customer-segments --task-type unsupervised \\
      --deployment none
        """
    )
    
    parser.add_argument(
        "--name", 
        required=True, 
        help="Project name (e.g., my-churn-model)"
    )
    parser.add_argument(
        "--output-dir", 
        default="..", 
        help="Parent directory for the new project (default: ..)"
    )
    parser.add_argument(
        "--task-type", 
        choices=["supervised", "unsupervised"], 
        default="supervised",
        help="ML task type (default: supervised)"
    )
    parser.add_argument(
        "--data-type", 
        choices=["tabular", "image", "database"], 
        default="tabular",
        help="Primary data type (default: tabular)"
    )
    parser.add_argument(
        "--deployment", 
        choices=["ray-serve", "ray-batch", "all", "none"], 
        default="all",
        help="Deployment target (default: all)"
    )
    
    return parser.parse_args()


def validate_project_name(name: str) -> None:
    """Validate project name."""
    if not name:
        raise ValueError("Project name cannot be empty")
    
    # Check for invalid characters
    invalid_chars = set(' <>:"/\\|?*')
    if any(c in name for c in invalid_chars):
        raise ValueError(
            f"Project name contains invalid characters. "
            f"Avoid: {' '.join(invalid_chars)}"
        )
    
    # Recommend kebab-case
    if '_' in name:
        logger.warning(
            f"Project name '{name}' uses underscores. "
            f"Consider kebab-case: '{name.replace('_', '-')}'"
        )


def create_directory_structure(root_path: Path) -> None:
    """Create the standard directory structure."""
    dirs = [
        "data/raw",
        "data/external",
        "data/processed",
        "scripts",
        "src/data_loaders",
        "src/deployment",
        "training",
        "notebooks",
    ]
    
    for d in dirs:
        try:
            (root_path / d).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {d}: {e}")
            raise
    
    # Create __init__.py files
    init_files = [
        "src/__init__.py",
        "src/data_loaders/__init__.py",
        "src/deployment/__init__.py",
    ]
    
    for init_file in init_files:
        try:
            (root_path / init_file).touch()
        except Exception as e:
            logger.error(f"Failed to create {init_file}: {e}")
            raise


def copy_file_safe(src: Path, dst: Path) -> bool:
    """
    Safely copy a file.
    
    Returns:
        True if successful, False otherwise
    """
    if not src.exists():
        logger.warning(f"Source file not found: {src}")
        return False
    
    try:
        # Create parent directory if needed
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        logger.debug(f"Copied {src.name} → {dst}")
        return True
    except Exception as e:
        logger.error(f"Failed to copy {src} to {dst}: {e}")
        return False


def copy_directory_safe(src: Path, dst: Path) -> bool:
    """
    Safely copy a directory recursively.
    
    Returns:
        True if successful, False otherwise
    """
    if not src.exists() or not src.is_dir():
        logger.warning(f"Source directory not found: {src}")
        return False
    
    try:
        shutil.copytree(src, dst, dirs_exist_ok=True)
        logger.debug(f"Copied directory {src.name} → {dst}")
        return True
    except Exception as e:
        logger.error(f"Failed to copy directory {src} to {dst}: {e}")
        return False


def render_template_safe(env: Environment, template_name: str, 
                         context: dict, output_path: Path) -> bool:
    """
    Safely render a Jinja2 template.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        template = env.get_template(template_name)
        content = template.render(**context)
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write(content)
        
        logger.info(f"✓ Rendered {template_name}")
        return True
        
    except TemplateNotFound:
        logger.error(f"Template not found: {template_name}")
        return False
    except Exception as e:
        logger.error(f"Failed to render {template_name}: {e}")
        return False


def scaffold_project(args):
    """Main scaffolding logic."""
    current_dir = Path.cwd()
    target_dir = Path(args.output_dir).resolve() / args.name
    
    # Validate
    validate_project_name(args.name)
    
    if target_dir.exists():
        logger.error(f"Target directory already exists: {target_dir}")
        logger.info("Please choose a different name or remove the existing directory")
        sys.exit(1)
    
    logger.info(f"Creating project '{args.name}' at {target_dir}")
    logger.info(f"  Task: {args.task_type}")
    logger.info(f"  Data: {args.data_type}")
    logger.info(f"  Deployment: {args.deployment}")
    
    # Create structure
    create_directory_structure(target_dir)
    
    # Setup Jinja2
    templates_dir = current_dir / "templates"
    if not templates_dir.exists():
        logger.error(f"Templates directory not found: {templates_dir}")
        sys.exit(1)
    
    env = Environment(loader=FileSystemLoader(templates_dir))
    
    # Context for templates
    context = {
        "project_name": args.name,
        "task_type": args.task_type,
        "data_type": args.data_type,
        "deployment": args.deployment
    }
    
    # Track success
    failed_operations = []
    
    # -------------------------------------------------------------------------
    # CORE FILES
    # -------------------------------------------------------------------------
    logger.info("\n→ Setting up core files...")
    
    core_files = [
        ("requirements.txt", "requirements.txt"),
        ("pyproject.toml", "pyproject.toml"),
        (".gitignore", ".gitignore"),
        (".pre-commit-config.yaml", ".pre-commit-config.yaml"),
        ("src/utils.py", "src/utils.py"),
    ]
    
    for src_name, dst_name in core_files:
        src = current_dir / src_name
        dst = target_dir / dst_name
        if not copy_file_safe(src, dst):
            failed_operations.append(f"Copy {src_name}")
    
    # Templates
    if not render_template_safe(env, "environment.yml.template", context, 
                                target_dir / "environment.yml"):
        failed_operations.append("Render environment.yml")
    
    if not render_template_safe(env, "setup.sh", context, 
                                target_dir / "setup.sh"):
        failed_operations.append("Render setup.sh")
    else:
        # Make executable
        try:
            os.chmod(target_dir / "setup.sh", 0o755)
        except Exception as e:
            logger.warning(f"Could not make setup.sh executable: {e}")
    
    # -------------------------------------------------------------------------
    # SCRIPTS
    # -------------------------------------------------------------------------
    logger.info("\n→ Setting up scripts...")
    
    scripts_dir = current_dir / "scripts"
    if scripts_dir.exists():
        for script in scripts_dir.glob("*.py"):
            if script.name != "scaffold_project.py":
                if not copy_file_safe(script, target_dir / "scripts" / script.name):
                    failed_operations.append(f"Copy script {script.name}")
    
    # -------------------------------------------------------------------------
    # DATA LOADERS
    # -------------------------------------------------------------------------
    logger.info(f"\n→ Setting up data loaders for {args.data_type}...")
    
    # Always copy base loader
    base_loader = current_dir / "src/data_loaders/base_loader.py"
    if not copy_file_safe(base_loader, target_dir / "src/data_loaders/base_loader.py"):
        failed_operations.append("Copy base_loader.py")
    
    # Copy specific loader
    loader_map = {
        "tabular": "tabular_loader.py",
        "image": "image_loader.py",
        "database": "database_loader.py"
    }
    
    loader_file = loader_map.get(args.data_type)
    if loader_file:
        src = current_dir / "src/data_loaders" / loader_file
        dst = target_dir / "src/data_loaders" / loader_file
        if not copy_file_safe(src, dst):
            failed_operations.append(f"Copy {loader_file}")
    
    # -------------------------------------------------------------------------
    # TRAINING SCRIPT
    # -------------------------------------------------------------------------
    logger.info(f"\n→ Setting up training for {args.task_type}...")
    
    train_template = f"train_{args.task_type}.py"
    src_template = templates_dir / train_template
    dst_train = target_dir / "training" / "train.py"
    
    if src_template.exists():
        if not copy_file_safe(src_template, dst_train):
            failed_operations.append(f"Copy {train_template}")
    else:
        logger.error(f"Training template not found: {train_template}")
        failed_operations.append(f"Find {train_template}")
    
    # Hyperparameter tuning
    tuning_src = templates_dir / "hyperparam_tuning.py"
    if tuning_src.exists():
        copy_file_safe(tuning_src, target_dir / "training" / "tune.py")
    
    # -------------------------------------------------------------------------
    # DEPLOYMENT
    # -------------------------------------------------------------------------
    if args.deployment != "none":
        logger.info(f"\n→ Setting up deployment for {args.deployment}...")
        
        if args.deployment in ["ray-serve", "all"]:
            serve_src = current_dir / "src/deployment/ray_serve.py"
            dockerfile = current_dir / "deployment/Dockerfile.ray"
            
            if not copy_file_safe(serve_src, target_dir / "src/deployment/ray_serve.py"):
                failed_operations.append("Copy ray_serve.py")
            if not copy_file_safe(dockerfile, target_dir / "Dockerfile.ray"):
                failed_operations.append("Copy Dockerfile.ray")
        
        if args.deployment in ["ray-batch", "all"]:
            batch_src = current_dir / "src/deployment/ray_batch.py"
            if not copy_file_safe(batch_src, target_dir / "src/deployment/ray_batch.py"):
                failed_operations.append("Copy ray_batch.py")
    
    # -------------------------------------------------------------------------
    # CONFIGURATION FILES
    # -------------------------------------------------------------------------
    logger.info("\n→ Rendering configuration files...")
    
    config_templates = [
        ("params.yaml.template", "params.yaml"),
        ("dvc.yaml.template", "dvc.yaml"),
        ("gitlab-ci.yml.template", ".gitlab-ci.yml"),
        ("Makefile.template", "Makefile"),
    ]
    
    for template_name, output_name in config_templates:
        if not render_template_safe(env, template_name, context, 
                                    target_dir / output_name):
            failed_operations.append(f"Render {template_name}")
    
    # -------------------------------------------------------------------------
    # README
    # -------------------------------------------------------------------------
    readme_content = """# {{ project_name }}

MLOps Project - Auto-generated

## Overview
- **Task Type**: {{ task_type }}
- **Data Type**: {{ data_type }}
- **Deployment**: {{ deployment }}

## Quick Start

### 1. Setup Environment
```bash
./setup.sh
```

### 2. Train Model
```bash
conda activate {{ project_name }}
dvc repro
```

### 3. View Results
```bash
mlflow ui
# Open http://localhost:5000
```

## Project Structure
```
├── data/               # Data files (tracked by DVC)
├── src/                # Source code
│   ├── data_loaders/   # Data loading utilities
│   └── deployment/     # Deployment code
├── training/           # Training scripts
├── scripts/            # MLOps automation scripts
├── notebooks/          # Jupyter notebooks
└── great_expectations/ # Data validation
```

## MLflow Tracking
- Experiment: `{{ project_name }}-exp`
- Model Registry: `{{ project_name }}-model`
- Tracking URI: `http://localhost:5001`

## CI/CD
This project uses GitLab CI/CD with:
- Automatic training on commits to main
- Champion vs Challenger comparison
- Auto-promotion when challenger is better

## Customization
1. Update `params.yaml` with your hyperparameters
2. Modify `training/train.py` with your model
3. Configure Great Expectations for data validation
4. Update `.gitlab-ci.yml` for your deployment needs

## Documentation
- [MLflow Docs](https://mlflow.org/docs/latest/index.html)
- [DVC Docs](https://dvc.org/doc)
- [Great Expectations Docs](https://docs.greatexpectations.io/)
"""
    
    readme_template = env.from_string(readme_content)
    with open(target_dir / "README.md", "w") as f:
        f.write(readme_template.render(**context))
    
    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("✅ PROJECT SCAFFOLDED SUCCESSFULLY")
    print("=" * 70)
    
    if failed_operations:
        print("\n⚠️  Some operations failed:")
        for op in failed_operations:
            print(f"  - {op}")
        print("\nProject created but may be incomplete. Check logs above.")
    
    print(f"\nProject location: {target_dir}")
    print("\nNext steps:")
    print(f"  1. cd {target_dir}")
    print("  2. ./setup.sh")
    print("  3. conda activate", args.name)
    print("  4. dvc repro")
    print("\nHappy MLOps! 🚀\n")


if __name__ == "__main__":
    try:
        args = parse_args()
        scaffold_project(args)
    except KeyboardInterrupt:
        logger.info("\nScaffolding cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nScaffolding failed: {e}")
        sys.exit(1)
