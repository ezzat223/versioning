#!/usr/bin/env python3
"""
MLOps Project Scaffolder
Generates a production-ready MLOps repository based on user specifications.
Uses Jinja2 for robust templating.
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Scaffold a new MLOps project")
    
    # Project Info
    parser.add_argument("--name", required=True, help="Project name (e.g., my-churn-model)")
    parser.add_argument("--output-dir", default="..", help="Parent directory for the new project")
    
    # Technical Decisions
    parser.add_argument("--task-type", choices=["supervised", "unsupervised"], default="supervised", 
                        help="Machine learning task type")
    parser.add_argument("--data-type", choices=["tabular", "image", "database"], default="tabular",
                        help="Primary data type")
    parser.add_argument("--deployment", choices=["ray-serve", "ray-batch", "all", "none"], default="all",
                        help="Deployment target")
    
    return parser.parse_args()

def create_directory_structure(root_path: Path):
    """Create the standard directory structure."""
    dirs = [
        "data",
        "scripts",
        "src",
        "src/data_loaders",
        "src/deployment",
        "templates",
        "training",
        "notebooks"
    ]
    
    for d in dirs:
        (root_path / d).mkdir(parents=True, exist_ok=True)
    
    # Create empty __init__.py files
    (root_path / "src" / "__init__.py").touch()
    (root_path / "src" / "data_loaders" / "__init__.py").touch()
    (root_path / "src" / "deployment" / "__init__.py").touch()

def copy_file(src: Path, dst: Path):
    """Copy a file if source exists."""
    if src.exists():
        shutil.copy2(src, dst)
    else:
        logger.warning(f"Source file not found: {src}")

def render_template(env, template_name, context, output_path):
    """Render a Jinja2 template to a file."""
    try:
        template = env.get_template(template_name)
        content = template.render(**context)
        with open(output_path, "w") as f:
            f.write(content)
        logger.info(f"Rendered {template_name} to {output_path}")
    except Exception as e:
        logger.error(f"Failed to render {template_name}: {e}")

def scaffold_project(args):
    current_dir = Path(os.getcwd())
    target_dir = Path(args.output_dir) / args.name
    
    if target_dir.exists():
        logger.error(f"Target directory {target_dir} already exists!")
        sys.exit(1)
        
    logger.info(f"Creating project '{args.name}' at {target_dir}...")
    create_directory_structure(target_dir)
    
    # Setup Jinja2 Environment
    env = Environment(loader=FileSystemLoader(current_dir / "templates"))
    
    # Context for templates
    context = {
        "project_name": args.name,
        "task_type": args.task_type,
        "data_type": args.data_type,
        "deployment": args.deployment
    }
    
    # -------------------------------------------------------------------------
    # 1. CORE FILES
    # -------------------------------------------------------------------------
    copy_file(current_dir / "requirements.txt", target_dir / "requirements.txt")
    copy_file(current_dir / "environment.yml", target_dir / "environment.yml")
    copy_file(current_dir / ".pre-commit-config.yaml", target_dir / ".pre-commit-config.yaml")
    copy_file(current_dir / "src/utils.py", target_dir / "src/utils.py")
    copy_file(current_dir / "src/core", target_dir / "src/core") # Directory copy if exists
    
    # Setup Script
    copy_file(current_dir / "templates/setup.sh", target_dir / "setup.sh")
    os.chmod(target_dir / "setup.sh", 0o755) # Make executable
    
    # Scripts
    for script in (current_dir / "scripts").glob("*.py"):
        if script.name != "scaffold_project.py": # Don't copy the scaffolder itself
            copy_file(script, target_dir / "scripts" / script.name)
            
    # -------------------------------------------------------------------------
    # 2. DATA LOADERS
    # -------------------------------------------------------------------------
    logger.info(f"Setting up data loaders for {args.data_type}...")
    
    # Always copy base loader
    copy_file(current_dir / "src/data_loaders/base_loader.py", target_dir / "src/data_loaders/base_loader.py")
    
    # Copy specific loader
    loader_map = {
        "tabular": "tabular_loader.py",
        "image": "image_loader.py",
        "database": "database_loader.py"
    }
    
    selected_loader = loader_map.get(args.data_type)
    if selected_loader:
        copy_file(current_dir / "src/data_loaders" / selected_loader, 
                 target_dir / "src/data_loaders" / selected_loader)
    
    # -------------------------------------------------------------------------
    # 3. TRAINING SCRIPT
    # -------------------------------------------------------------------------
    logger.info(f"Setting up training script for {args.task_type}...")
    
    template_name = f"train_{args.task_type}.py"
    src_template = current_dir / "templates" / template_name
    dst_train = target_dir / "training" / "train.py"
    
    if src_template.exists():
        copy_file(src_template, dst_train)
    else:
        logger.error(f"Template {template_name} not found!")
        
    # Copy other templates to templates/ dir for reference
    for t in (current_dir / "templates").glob("*"):
        if not t.name.endswith(".template"): # Don't copy .template files to templates dir
            copy_file(t, target_dir / "templates" / t.name)

    # -------------------------------------------------------------------------
    # 4. DEPLOYMENT
    # -------------------------------------------------------------------------
    logger.info(f"Setting up deployment for {args.deployment}...")
    
    copy_file(current_dir / "src/deployment/manager.py", target_dir / "src/deployment/manager.py")
    copy_file(current_dir / "src/deployment/deployment_configs.yaml", target_dir / "src/deployment/deployment_configs.yaml")
    
    if args.deployment in ["ray-serve", "all"]:
        copy_file(current_dir / "src/deployment/ray_serve.py", target_dir / "src/deployment/ray_serve.py")
        copy_file(current_dir / "deployment/Dockerfile.ray", target_dir / "Dockerfile.ray")
        
    if args.deployment in ["ray-batch", "all"]:
        copy_file(current_dir / "src/deployment/ray_batch.py", target_dir / "src/deployment/ray_batch.py")

    # -------------------------------------------------------------------------
    # 5. RENDER TEMPLATES
    # -------------------------------------------------------------------------
    logger.info("Rendering configuration files...")
    
    render_template(env, "params.yaml.template", context, target_dir / "params.yaml")
    render_template(env, "dvc.yaml.template", context, target_dir / "dvc.yaml")
    render_template(env, "gitlab-ci.yml.template", context, target_dir / ".gitlab-ci.yml")
    
    # README (Inline template)
    readme_content = """# {{ project_name }}

Generated MLOps Project.

## Overview
- **Task**: {{ task_type }}
- **Data**: {{ data_type }}
- **Deployment**: {{ deployment }}

## Setup
1. `./setup.sh` - Initializes Conda env, DVC, and Git hooks.

## Training
Run: `dvc repro`
"""
    # Use Jinja to render inline string
    readme_template = env.from_string(readme_content)
    with open(target_dir / "README.md", "w") as f:
        f.write(readme_template.render(**context))

    logger.info(f"✓ Project scaffolded successfully at {target_dir}")
    logger.info("Next steps:")
    logger.info(f"  cd {target_dir}")
    logger.info("  ./setup.sh")

if __name__ == "__main__":
    args = parse_args()
    scaffold_project(args)
