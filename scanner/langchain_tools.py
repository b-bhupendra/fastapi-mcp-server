"""
LangChain Tools & MCP Bridge Module.
Exposes modularized scanner, objective extractor, and Pinecone RAG
capabilities as first-class LangChain @tool objects for local agents and MCP servers.
"""

import os
import json
from typing import List, Optional
from langchain_core.tools import tool

from scanner.git_syncer import inspect_git_repository, get_git_sync_summary
from scanner.objective_extractor import extract_project_objective, call_ollama_embed
from scanner.pinecone_store import query_vectors, PINECONE_INDEX_NAME

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(ROOT_DIR, "data", "projects_cache.json")

def load_cached_catalog():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r") as f:
                return json.load(f).get("projects", [])
        except Exception:
            pass
    return []

@tool
def get_project_objective_tool(project_name: str) -> str:
    """
    Retrieve specifically what a repository or project was trying to achieve.
    Returns the intended objective, problem statement, development status,
    and recommended next steps to complete it.
    """
    projects = load_cached_catalog()
    for p in projects:
        if p.get("project_name", "").lower() == project_name.lower():
            obj = p.get("intended_objective", "Objective not yet cataloged.")
            status = p.get("project_status", "In Development")
            steps = p.get("next_steps", ["Continue development", "Review documentation"])
            steps_str = "\n".join([f"  - {s}" for s in steps])
            return f"""### 🎯 Project Objective: `{p['project_name']}`
- **Intended Goal / Problem Statement**:
  {obj}
- **Current Lifecycle Status**: `{status}`
- **Primary Tech Stack**: {', '.join(p.get('tech_stack', []))}
- **Recommended Next Steps to Achieve Objective**:
{steps_str}
"""
    return f"Project '{project_name}' was not found in the local intelligence catalog. Run a project scan first."

@tool
def check_github_sync_tool(project_name: str = "") -> str:
    """
    Check if local repositories have uncommitted changes or unpushed commits
    relative to their GitHub origin remotes, and provides sync recommendations.
    """
    base_dir = os.environ.get("SCAN_TARGET_DIR", "/home/bhupendra/.gemini/antigravity-ide/scratch")
    if project_name:
        target_path = os.path.join(base_dir, project_name)
        if not os.path.isdir(target_path):
            return f"Project directory '{project_name}' not found."
        info = inspect_git_repository(target_path)
        status = "✅ Clean" if info["is_synced"] else "⚠️ Unsynced"
        return f"""### 🐙 GitHub Sync Status for `{project_name}` [{status}]
- **Branch**: `{info['git_branch']}`
- **GitHub Remote**: {info['git_remote']}
- **Unpushed Commits**: {info['unpushed_commits']}
- **Dirty Files**: {info['dirty_files_count']}
- **Action Needed**: {info['sync_recommendation']}
"""
    else:
        # Check all repos
        candidates = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        summaries = get_git_sync_summary(candidates)
        dirty_items = [s for s in summaries if not s["is_synced"]]
        if not dirty_items:
            return "🎉 All local project repositories are clean and fully synced with GitHub!"
        out = ["### ⚠️ Unsynced Repositories Requiring Attention:\n"]
        for s in dirty_items:
            out.append(f"- **`{s['project_name']}`** (`{s['git_branch']}`): {s['sync_recommendation']}")
        return "\n".join(out)

@tool
def query_codebase_rag_tool(query: str, top_k: int = 5) -> str:
    """
    Perform semantic vector RAG similarity search across Pinecone DB for relevant
    codebases, architecture patterns, objectives, and code chunks.
    """
    q_vec = call_ollama_embed(query)
    matches = query_vectors(q_vec, top_k=top_k)
    if not matches:
        return f"No Pinecone vector matches found for: '{query}'."

    res = [f"### 🔍 Pinecone Semantic RAG Matches for '{query}':\n"]
    for m in matches:
        meta = m.get("metadata", {})
        score = m.get("score", 0.0)
        res.append(f"#### 📁 `{meta.get('project_name')}` (Similarity: {score:.3f})")
        res.append(f"- **Intended Objective**: {meta.get('intended_objective', 'N/A')}")
        res.append(f"- **Tech Stack**: {meta.get('tech_stack', 'N/A')}")
        res.append(f"- **Summary**: {meta.get('consensus_summary', 'N/A')}")
        res.append("")
    return "\n".join(res)

@tool
def read_project_file_tool(project_name: str, file_path: str, max_lines: int = 100) -> str:
    """
    Read the actual file contents from a local project repository for code inspection.
    """
    base_dir = os.environ.get("SCAN_TARGET_DIR", "/home/bhupendra/.gemini/antigravity-ide/scratch")
    target_file = os.path.join(base_dir, project_name, file_path)
    if not os.path.isfile(target_file):
        return f"File '{file_path}' in project '{project_name}' does not exist on disk."
    try:
        with open(target_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = [f.readline() for _ in range(max_lines)]
            return f"### 📄 `{project_name}/{file_path}`\n```\n{''.join(lines)}\n```"
    except Exception as e:
        return f"Error reading file: {e}"

def get_all_langchain_tools() -> List:
    """Returns list of LangChain tools ready to be bound to an agent or LLM."""
    return [
        get_project_objective_tool,
        check_github_sync_tool,
        query_codebase_rag_tool,
        read_project_file_tool
    ]
