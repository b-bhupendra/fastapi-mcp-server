"""
Git Syncer & Inspector Module.
Handles git status, unpushed commits, dirty files, remote URL sanitization,
and GitHub sync state checks.
"""

import os
import re
import subprocess
from typing import Dict, Any, Optional, List

def run_git_cmd(cmd: List[str], cwd: str) -> Optional[str]:
    try:
        res = subprocess.run(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None

def sanitize_remote_url(raw_url: str) -> str:
    """Strips embedded user credentials and tokens from git remote URLs."""
    if not raw_url:
        return ""
    # Strip user:token@ or token@ patterns
    return re.sub(r"https://[^@]+@", "https://", raw_url)

def inspect_git_repository(repo_path: str) -> Dict[str, Any]:
    git_dir = os.path.join(repo_path, ".git")
    if not os.path.isdir(git_dir):
        return {
            "is_git_repo": False,
            "git_branch": "none",
            "git_remote": "",
            "is_synced": True,
            "unpushed_commits": 0,
            "dirty_files_count": 0,
            "last_commit_date": "",
            "sync_recommendation": "Not a git repository."
        }

    branch = run_git_cmd(["git", "branch", "--show-current"], cwd=repo_path) or "main"
    raw_remote = run_git_cmd(["git", "remote", "get-url", "origin"], cwd=repo_path) or ""
    remote = sanitize_remote_url(raw_remote)

    dirty_out = run_git_cmd(["git", "status", "--porcelain"], cwd=repo_path)
    dirty_count = len(dirty_out.splitlines()) if dirty_out else 0

    unpushed_out = run_git_cmd(["git", "rev-list", "@{u}..HEAD", "--count"], cwd=repo_path)
    unpushed_count = 0
    if unpushed_out and unpushed_out.isdigit():
        unpushed_count = int(unpushed_out)

    last_commit_date = run_git_cmd(["git", "log", "-1", "--format=%cI"], cwd=repo_path) or ""
    is_synced = (dirty_count == 0 and unpushed_count == 0)

    # Sync recommendations
    if is_synced:
        recommendation = "Fully synchronized with GitHub remote."
    elif dirty_count > 0 and unpushed_count > 0:
        recommendation = f"Commit {dirty_count} modified files, then push {unpushed_count} unpushed commits to origin/{branch}."
    elif dirty_count > 0:
        recommendation = f"Commit {dirty_count} modified files to branch '{branch}'."
    else:
        recommendation = f"Push {unpushed_count} commits to origin/{branch}."

    return {
        "is_git_repo": True,
        "git_branch": branch,
        "git_remote": remote,
        "is_synced": is_synced,
        "unpushed_commits": unpushed_count,
        "dirty_files_count": dirty_count,
        "last_commit_date": last_commit_date,
        "sync_recommendation": recommendation
    }

def get_git_sync_summary(repo_paths: List[str]) -> List[Dict[str, Any]]:
    summaries = []
    for p in repo_paths:
        info = inspect_git_repository(p)
        info["project_name"] = os.path.basename(p)
        summaries.append(info)
    return summaries
