"""
Utility functions for Git metadata extraction and MLflow tagging.
"""
from pathlib import Path
from typing import Dict, Optional
import git
import mlflow
import pandas as pd

def get_git_metadata(repo_path: Optional[str] = None) -> Dict[str, str]:
    """
    Extract Git metadata from the current repository.
    
    Args:
        repo_path: Path to git repository. If None, uses current directory.
    
    Returns:
        Dictionary containing git metadata (commit_sha, branch, repo_name, is_dirty)
    """
    if repo_path is None:
        repo_path = Path(__file__).parent.parent
    
    try:
        repo = git.Repo(repo_path, search_parent_directories=True)
        
        # Get current commit SHA
        commit_sha = repo.head.commit.hexsha
        
        # Get current branch name
        try:
            branch = repo.active_branch.name
        except TypeError:
            # Detached HEAD state
            branch = "detached-head"
        
        # Get repository name from remote or directory
        try:
            remote_url = repo.remotes.origin.url
            repo_name = remote_url.split('/')[-1].replace('.git', '')
        except (AttributeError, IndexError):
            repo_name = Path(repo.working_dir).name
        
        # Check if working directory is dirty
        is_dirty = repo.is_dirty()
        
        return {
            "git.commit_sha": commit_sha,
            "git.branch": branch,
            "git.repo_name": repo_name,
            "git.is_dirty": str(is_dirty),
            "git.commit_message": repo.head.commit.message.strip(),
        }
    
    except git.InvalidGitRepositoryError:
        print("Warning: Not a git repository. Git metadata will not be available.")
        return {
            "git.commit_sha": "unknown",
            "git.branch": "unknown",
            "git.repo_name": "unknown",
            "git.is_dirty": "unknown",
            "git.commit_message": "unknown",
        }


def validate_git_state(metadata: Dict[str, str], strict: bool = False) -> bool:
    """
    Validate that the git repository is in a clean state.
    
    Args:
        metadata: Git metadata dictionary
        strict: If True, raise error on dirty state. If False, just warn.
    
    Returns:
        True if repository is clean, False otherwise
    """
    is_dirty = metadata.get("git.is_dirty", "false") == "True"
    
    if is_dirty:
        warning_msg = (
            "⚠️  WARNING: Working directory has uncommitted changes!\n"
            "For full reproducibility, commit all changes before training."
        )
        print(warning_msg)
        
        if strict:
            raise RuntimeError("Repository has uncommitted changes. Commit before proceeding.")
        
        return False
    
    return True

def print_git_info(metadata: Dict[str, str]) -> None:
    """Pretty print git metadata."""
    print("\n" + "="*60)
    print("GIT METADATA")
    print("="*60)
    for key, value in metadata.items():
        print(f"{key:30s}: {value}")
    print("="*60 + "\n")
