#!/usr/bin/env python3
"""
Local Project Intelligence Scanner with Dual-Ollama Consensus & Central Pinecone DB Ingestion.

Modular Architecture:
- scanner.git_syncer: Git state, unpushed commits, dirty files, GitHub remotes.
- scanner.objective_extractor: Dual-Ollama multi-temperature consensus extracting what the project was trying to achieve.
- scanner.pinecone_store: Serverless vector database indexing and metadata upsert.
- scanner.langchain_tools: LangChain @tool bridge for local agents.
"""

import os
import sys
import json
import datetime
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import urllib.request
import urllib.error

# Auto-load .env file if present
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

env_file = os.path.join(root_dir, ".env")
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v.strip("\"'"))

from scanner.git_syncer import inspect_git_repository
from scanner.objective_extractor import extract_project_objective, call_ollama_embed, MODELS
from scanner.pinecone_store import upsert_projects, PINECONE_INDEX_NAME

VERCEL_MCP_URL = os.environ.get("VERCEL_MCP_URL", "https://fastapi-mcp-server.vercel.app")
SYNC_SECRET = os.environ.get("SYNC_SECRET", "dev-sync-key")

# Try importing Rich for elite terminal UI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TaskProgressColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

def log_info(msg: str):
    if RICH_AVAILABLE:
        console.print(f"[cyan]ℹ[/cyan] {msg}")
    else:
        print(f"[*] {msg}")

def log_success(msg: str):
    if RICH_AVAILABLE:
        console.print(f"[bold green]✔[/bold green] {msg}")
    else:
        print(f"[+] {msg}")

def log_warn(msg: str):
    if RICH_AVAILABLE:
        console.print(f"[bold yellow]⚠[/bold yellow] {msg}")
    else:
        print(f"[!] {msg}")

def log_error(msg: str):
    if RICH_AVAILABLE:
        console.print(f"[bold red]✖[/bold red] {msg}")
    else:
        print(f"[-] {msg}")

def print_header():
    if RICH_AVAILABLE:
        header_text = (
            "[bold cyan]Project Intelligence Scanner & Pinecone Ingester[/bold cyan]\n"
            f"[dim]Dual-Ollama Consensus ({', '.join(MODELS)}) | 768-dim Embeddings[/dim]\n"
            f"[dim]Central Cloud Storage: Pinecone DB Index: [bold]{PINECONE_INDEX_NAME}[/bold][/dim]"
        )
        console.print(Panel(header_text, box=box.ROUNDED, border_style="cyan", expand=False))
    else:
        print("=" * 60)
        print("Project Intelligence Scanner & Pinecone Ingester")
        print(f"Models: {', '.join(MODELS)} | Storage: Pinecone ({PINECONE_INDEX_NAME})")
        print("=" * 60)

def extract_project_context(repo_path: str) -> Dict[str, Any]:
    project_name = os.path.basename(os.path.abspath(repo_path))
    readme_content = ""
    for r in ["README.md", "README.rst", "README.txt", "readme.md"]:
        rpath = os.path.join(repo_path, r)
        if os.path.isfile(rpath):
            try:
                with open(rpath, "r", encoding="utf-8", errors="ignore") as f:
                    readme_content = f.read(2500)
                break
            except Exception:
                pass

    manifest_info = {}
    primary_language = "Unknown"
    tech_stack = []

    if os.path.exists(os.path.join(repo_path, "requirements.txt")):
        primary_language = "Python"
        tech_stack.append("Python")
        try:
            with open(os.path.join(repo_path, "requirements.txt"), "r") as f:
                manifest_info["requirements"] = f.read(500)
        except Exception:
            pass

    if os.path.exists(os.path.join(repo_path, "package.json")):
        primary_language = "TypeScript/JavaScript"
        tech_stack.append("Node.js")
        try:
            with open(os.path.join(repo_path, "package.json"), "r") as f:
                pkg_data = json.load(f)
                manifest_info["dependencies"] = list(pkg_data.get("dependencies", {}).keys())
                tech_stack.extend(manifest_info["dependencies"][:5])
        except Exception:
            pass

    if os.path.exists(os.path.join(repo_path, "go.mod")):
        primary_language = "Go"
        tech_stack.append("Go")

    if os.path.exists(os.path.join(repo_path, "Cargo.toml")):
        primary_language = "Rust"
        tech_stack.append("Rust")

    top_files = []
    try:
        for item in os.listdir(repo_path):
            if not item.startswith("."):
                top_files.append(item)
    except Exception:
        pass

    return {
        "project_name": project_name,
        "readme_preview": readme_content[:1500],
        "primary_language": primary_language,
        "tech_stack": list(set(tech_stack)),
        "manifest_info": manifest_info,
        "top_files": top_files[:10]
    }

def calculate_priority_score(git_info: Dict[str, Any]) -> int:
    score = 50
    if not git_info.get("is_synced", True):
        score += 25
    score += min(git_info.get("unpushed_commits", 0) * 5, 20)
    score += min(git_info.get("dirty_files_count", 0) * 2, 10)
    return min(max(score, 1), 100)

def scan_repository(repo_path: str, progress_cb=None) -> Tuple[Dict[str, Any], List[float]]:
    git_info = inspect_git_repository(repo_path)
    ctx = extract_project_context(repo_path)
    
    # Extract intended objective & status using Dual-Ollama consensus
    obj_info = extract_project_objective(ctx, progress_cb=progress_cb)
    priority = calculate_priority_score(git_info)

    intended_objective = obj_info["intended_objective"]
    project_status = obj_info["project_status"]
    next_steps = obj_info["next_steps"]

    vector_text = f"Project: {ctx['project_name']}. Objective: {intended_objective}. Stack: {', '.join(ctx['tech_stack'])}"
    embedding = call_ollama_embed(vector_text)

    proj_data = {
        "project_name": ctx["project_name"],
        "primary_language": ctx["primary_language"],
        "tech_stack": ctx["tech_stack"],
        "git_remote": git_info["git_remote"],
        "git_branch": git_info["git_branch"],
        "is_synced": git_info["is_synced"],
        "unpushed_commits": git_info["unpushed_commits"],
        "dirty_files_count": git_info["dirty_files_count"],
        "last_commit_date": git_info["last_commit_date"],
        "priority_score": priority,
        "intended_objective": intended_objective,
        "project_status": project_status,
        "next_steps": next_steps,
        "consensus_summary": intended_objective,
        "key_files": ctx["top_files"],
        "embedding_dim": len(embedding),
        "last_scanned": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "chunks": [
            {"id": f"{ctx['project_name']}:overview", "text": intended_objective, "type": "intended_objective"},
            {"id": f"{ctx['project_name']}:readme", "text": ctx["readme_preview"][:800], "type": "readme_chunk"}
        ]
    }
    return proj_data, embedding

def sync_to_vercel_mcp(projects: List[Dict[str, Any]]) -> bool:
    endpoint = f"{VERCEL_MCP_URL}/api/sync"
    try:
        payload = json.dumps({"projects": projects}).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-Sync-Key": SYNC_SECRET
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return True
    except Exception:
        return False

def render_summary_table(projects: List[Dict[str, Any]], pinecone_ok: bool):
    if not RICH_AVAILABLE:
        print("\n" + "=" * 80)
        print(f"{'PROJECT':<22} {'LANG':<10} {'SYNC':<10} {'STATUS':<15} {'OBJECTIVE':<30}")
        print("-" * 80)
        for p in projects:
            sync_txt = "CLEAN" if p['is_synced'] else "DIRTY"
            print(f"{p['project_name']:<22} {p['primary_language']:<10} {sync_txt:<10} {p.get('project_status', 'WIP'):<15} {p.get('intended_objective', '')[:28]}...")
        print("=" * 80)
        return

    table = Table(
        title="✨ Scanned Projects Intelligence & Intended Objectives",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold green",
        show_lines=True
    )
    table.add_column("Repository", style="bold white", width=20)
    table.add_column("Git Sync", justify="center", width=12)
    table.add_column("Lifecycle", style="yellow", width=14)
    table.add_column("Intended Objective / Goal", style="white", width=42)
    table.add_column("Pinecone", justify="center", width=10)

    for p in projects:
        if p["is_synced"]:
            sync_badge = "[bold green]🟢 Synced[/bold green]"
        elif p["unpushed_commits"] > 0:
            sync_badge = f"[bold yellow]🟡 {p['unpushed_commits']} unpushed[/bold yellow]"
        else:
            sync_badge = f"[bold red]🔴 {p['dirty_files_count']} dirty[/bold red]"

        obj_preview = p.get("intended_objective", "")
        if len(obj_preview) > 90:
            obj_preview = obj_preview[:87] + "..."

        pinecone_badge = "[bold green]✔ Done[/bold green]" if pinecone_ok else "[dim]Cached[/dim]"

        table.add_row(
            p["project_name"],
            sync_badge,
            p.get("project_status", "In Development"),
            obj_preview,
            pinecone_badge
        )

    console.print(table)

def main():
    print_header()
    target_dirs = sys.argv[1:] if len(sys.argv) > 1 else ["/home/bhupendra/.gemini/antigravity-ide/scratch"]

    candidate_paths = []
    for base_dir in target_dirs:
        if not os.path.exists(base_dir):
            continue
        if os.path.isdir(os.path.join(base_dir, ".git")):
            candidate_paths.append(base_dir)
            continue
        for item in sorted(os.listdir(base_dir)):
            full_path = os.path.join(base_dir, item)
            if os.path.isdir(full_path) and not item.startswith("."):
                candidate_paths.append(full_path)

    if not candidate_paths:
        log_warn("No project directories found to scan.")
        return

    log_info(f"Discovered [bold]{len(candidate_paths)}[/bold] candidate repositories across: {', '.join(target_dirs)}")

    all_projects = []
    all_embeddings = []

    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn("dots", style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=35, style="dim white", complete_style="bold green"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            total_task = progress.add_task("[bold cyan]Scanning Repositories...", total=len(candidate_paths))
            sub_task = progress.add_task("[dim]Analyzing objectives...", total=6)

            for repo_path in candidate_paths:
                repo_name = os.path.basename(repo_path)
                progress.update(total_task, description=f"[bold cyan]Analyzing Objective:[/bold cyan] [yellow]{repo_name}[/yellow]")
                progress.reset(sub_task)

                def update_sub_task(current, total, model_info):
                    progress.update(
                        sub_task,
                        total=total,
                        completed=current,
                        description=f"  [dim cyan]Dual-Ollama:[/dim cyan] [white]{model_info}[/white]"
                    )

                proj_data, embedding = scan_repository(repo_path, progress_cb=update_sub_task)
                all_projects.append(proj_data)
                all_embeddings.append(embedding)

                progress.advance(total_task)

            progress.update(sub_task, visible=False)
    else:
        for idx, repo_path in enumerate(candidate_paths, 1):
            repo_name = os.path.basename(repo_path)
            print(f"[{idx}/{len(candidate_paths)}] Analyzing {repo_name}...")
            proj_data, embedding = scan_repository(repo_path)
            all_projects.append(proj_data)
            all_embeddings.append(embedding)

    # Save to local cache
    cache_path = os.path.join(root_dir, "data", "projects_cache.json")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"projects": all_projects}, f, indent=2)
    log_success(f"Saved local cache to [underline]{cache_path}[/underline]")

    # Pinecone Upsert
    if RICH_AVAILABLE:
        with console.status("[bold cyan]Upserting project vectors & objectives to Pinecone DB...", spinner="aesthetic"):
            pinecone_ok = upsert_projects(all_projects, all_embeddings)
    else:
        pinecone_ok = upsert_projects(all_projects, all_embeddings)

    if pinecone_ok:
        log_success(f"Successfully upserted [bold green]{len(all_projects)}[/bold green] vectors to Pinecone Index ([cyan]{PINECONE_INDEX_NAME}[/cyan])!")
    else:
        log_warn("Pinecone DB upsert completed in offline/cache mode.")

    # Vercel Sync
    sync_to_vercel_mcp(all_projects)

    # Render Final Rich Dashboard Table
    console.print()
    render_summary_table(all_projects, pinecone_ok)

if __name__ == "__main__":
    main()
