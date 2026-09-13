#!/usr/bin/env python3
"""
Local Project Intelligence Scanner with Dual-Ollama Consensus & Central Pinecone DB Ingestion.

Features:
- Beautiful Rich CLI terminal interface with animated progress bars, live spinners, and summary tables.
- Scans local directories for git repositories and projects.
- Inspects git sync state (dirty files, unpushed commits, active branch, remotes).
- Extracts READMEs, package configs, entry points.
- Dual-Ollama multi-temperature consensus (qwen2.5:7b + llama3.2:latest at T=0.0, 0.5, 0.9).
- Vectorizes with nomic-embed-text (768-dim) and upserts directly to Pinecone DB.
- Syncs with Vercel FastAPI MCP server.
"""

import os
import sys
import json
import glob
import subprocess
import datetime
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import urllib.request
import urllib.error

# Auto-load .env file if present
env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v.strip("\"'"))

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "project-intelligence")
VERCEL_MCP_URL = os.environ.get("VERCEL_MCP_URL", "https://fastapi-mcp-server.vercel.app")
SYNC_SECRET = os.environ.get("SYNC_SECRET", "dev-sync-key")

MODELS = ["qwen2.5:7b", "llama3.2:latest"]
TEMPERATURES = [0.0, 0.5, 0.9]

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
        TimeRemainingColumn,
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

def run_cmd(cmd: List[str], cwd: str) -> Optional[str]:
    try:
        res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None

def inspect_git_status(repo_path: str) -> Dict[str, Any]:
    git_dir = os.path.join(repo_path, ".git")
    is_git = os.path.isdir(git_dir)
    if not is_git:
        return {
            "is_git_repo": False,
            "git_branch": "none",
            "git_remote": "",
            "is_synced": True,
            "unpushed_commits": 0,
            "dirty_files_count": 0,
            "last_commit_date": ""
        }

    branch = run_cmd(["git", "branch", "--show-current"], cwd=repo_path) or "main"
    raw_remote = run_cmd(["git", "remote", "get-url", "origin"], cwd=repo_path) or ""
    import re
    remote = re.sub(r"https://[^@]+@", "https://", raw_remote)
    dirty_out = run_cmd(["git", "status", "--porcelain"], cwd=repo_path)
    dirty_count = len(dirty_out.splitlines()) if dirty_out else 0

    unpushed_out = run_cmd(["git", "rev-list", "@{u}..HEAD", "--count"], cwd=repo_path)
    unpushed_count = 0
    if unpushed_out and unpushed_out.isdigit():
        unpushed_count = int(unpushed_out)

    last_commit_date = run_cmd(["git", "log", "-1", "--format=%cI"], cwd=repo_path) or ""
    is_synced = (dirty_count == 0 and unpushed_count == 0)

    return {
        "is_git_repo": True,
        "git_branch": branch,
        "git_remote": remote,
        "is_synced": is_synced,
        "unpushed_commits": unpushed_count,
        "dirty_files_count": dirty_count,
        "last_commit_date": last_commit_date
    }

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

    # Detect package files
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

    # List top files
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

def call_ollama_generate(model: str, prompt: str, temperature: float) -> str:
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 120
        }
    }
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=40) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except Exception as e:
        return ""

def call_ollama_embed(text: str) -> List[float]:
    url = f"{OLLAMA_URL}/api/embeddings"
    payload = {
        "model": "nomic-embed-text",
        "prompt": text[:2000]
    }
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("embedding", [0.0] * 768)
    except Exception as e:
        return [0.0] * 768

def run_dual_ollama_consensus(ctx: Dict[str, Any], progress_cb=None) -> str:
    prompt = f"""You are an expert code intelligence evaluator. Summarize the following project concisely in 2 sentences.
Focus strictly on the actual purpose, technology stack, and architecture shown in the files. Do not invent features.

Project Name: {ctx['project_name']}
Primary Language: {ctx['primary_language']}
Files: {', '.join(ctx['top_files'])}
Tech Stack: {', '.join(ctx['tech_stack'])}
README Preview:
{ctx['readme_preview'][:1000]}
"""
    responses = []
    total_passes = len(MODELS) * len(TEMPERATURES)
    current_pass = 0

    for model in MODELS:
        for temp in TEMPERATURES:
            current_pass += 1
            if progress_cb:
                progress_cb(current_pass, total_passes, f"{model} (T={temp})")
            out = call_ollama_generate(model, prompt, temp)
            if out:
                responses.append(out)

    if not responses:
        return f"{ctx['project_name']} is a {ctx['primary_language']} project containing {', '.join(ctx['top_files'][:5])}."

    # Consensus Selection
    best_candidate = responses[0]
    best_score = -1
    keywords = set(ctx['top_files'] + ctx['tech_stack'] + [ctx['project_name'].lower()])

    for cand in responses:
        cand_lower = cand.lower()
        score = sum(1 for kw in keywords if kw.lower() in cand_lower)
        if "here is" in cand_lower or "based on" in cand_lower:
            score -= 1
        if score > best_score:
            best_score = score
            best_candidate = cand

    clean_summary = best_candidate.replace("Here is a 2-sentence summary:", "").strip()
    return clean_summary

def calculate_priority_score(git_info: Dict[str, Any]) -> int:
    score = 50
    if not git_info.get("is_synced", True):
        score += 25
    score += min(git_info.get("unpushed_commits", 0) * 5, 20)
    score += min(git_info.get("dirty_files_count", 0) * 2, 10)
    return min(max(score, 1), 100)

def upsert_to_pinecone(projects: List[Dict[str, Any]], embeddings: List[List[float]]) -> bool:
    """Upserts projects and embeddings to Pinecone DB."""
    if not PINECONE_API_KEY:
        log_warn("PINECONE_API_KEY not set; skipping direct cloud vector upsert.")
        return False
    try:
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        active_indexes = [i.name for i in pc.list_indexes()]
        if PINECONE_INDEX_NAME not in active_indexes:
            log_info(f"Creating Pinecone index: {PINECONE_INDEX_NAME} (dimension=768)...")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
                time.sleep(2)
        
        index = pc.Index(PINECONE_INDEX_NAME)
        vectors_to_upsert = []
        for p, emb in zip(projects, embeddings):
            vec_id = f"proj:{p['project_name']}:overview"
            metadata = {
                "project_name": p["project_name"],
                "primary_language": p["primary_language"],
                "is_synced": p["is_synced"],
                "unpushed_commits": p["unpushed_commits"],
                "dirty_files_count": p["dirty_files_count"],
                "priority_score": p["priority_score"],
                "consensus_summary": p["consensus_summary"][:1000],
                "tech_stack": p["tech_stack"],
                "git_remote": p["git_remote"]
            }
            vectors_to_upsert.append({
                "id": vec_id,
                "values": emb,
                "metadata": metadata
            })

        index.upsert(vectors=vectors_to_upsert)
        return True
    except Exception as e:
        log_error(f"Pinecone upsert error: {e}")
        return False

def sync_to_vercel_mcp(projects: List[Dict[str, Any]]) -> bool:
    """Syncs project profiles to the Vercel FastAPI MCP server."""
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
            res = json.loads(resp.read().decode("utf-8"))
            return True
    except Exception:
        return False

def render_summary_table(projects: List[Dict[str, Any]], pinecone_ok: bool):
    if not RICH_AVAILABLE:
        print("\n" + "=" * 80)
        print(f"{'PROJECT':<25} {'LANG':<12} {'SYNC':<10} {'UNPUSHED':<10} {'PRIORITY':<10}")
        print("-" * 80)
        for p in projects:
            sync_txt = "CLEAN" if p['is_synced'] else "DIRTY"
            print(f"{p['project_name']:<25} {p['primary_language']:<12} {sync_txt:<10} {p['unpushed_commits']:<10} {p['priority_score']}/100")
        print("=" * 80)
        return

    table = Table(
        title="✨ Scanned Projects Intelligence Catalog",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold green",
        show_lines=True
    )
    table.add_column("Repository", style="bold white", width=22)
    table.add_column("Tech Stack", style="cyan", width=20)
    table.add_column("Branch", style="dim", width=10)
    table.add_column("Git Sync", justify="center", width=12)
    table.add_column("Unpushed", justify="center", width=10)
    table.add_column("Priority", justify="center", width=12)
    table.add_column("Pinecone RAG", justify="center", width=14)

    for p in projects:
        # Git Sync Badge
        if p["is_synced"]:
            sync_badge = "[bold green]🟢 Synced[/bold green]"
        elif p["unpushed_commits"] > 0:
            sync_badge = f"[bold yellow]🟡 {p['unpushed_commits']} unpushed[/bold yellow]"
        else:
            sync_badge = f"[bold red]🔴 {p['dirty_files_count']} dirty[/bold red]"

        # Priority Color Coding
        score = p["priority_score"]
        if score >= 75:
            p_text = f"[bold red]{score}/100 ⚡[/bold red]"
        elif score >= 50:
            p_text = f"[bold yellow]{score}/100[/bold yellow]"
        else:
            p_text = f"[bold green]{score}/100[/bold green]"

        pinecone_badge = "[bold green]✔ Ingested[/bold green]" if pinecone_ok else "[dim]Cached[/dim]"
        stack_str = ", ".join(p["tech_stack"][:3]) if p["tech_stack"] else p["primary_language"]

        table.add_row(
            p["project_name"],
            stack_str,
            p["git_branch"],
            sync_badge,
            str(p["unpushed_commits"]),
            p_text,
            pinecone_badge
        )

    console.print(table)

def main():
    print_header()
    target_dirs = sys.argv[1:] if len(sys.argv) > 1 else ["/home/bhupendra/.gemini/antigravity-ide/scratch"]

    # Discover candidate project directories
    candidate_paths = []
    for base_dir in target_dirs:
        if not os.path.exists(base_dir):
            continue
        # If target directory is itself a git repository, scan it directly
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
            sub_task = progress.add_task("[dim]Initializing consensus...", total=6)

            for repo_path in candidate_paths:
                repo_name = os.path.basename(repo_path)
                progress.update(total_task, description=f"[bold cyan]Scanning:[/bold cyan] [yellow]{repo_name}[/yellow]")
                progress.reset(sub_task)

                git_info = inspect_git_status(repo_path)
                ctx = extract_project_context(repo_path)

                def update_sub_task(current, total, model_info):
                    progress.update(
                        sub_task,
                        total=total,
                        completed=current,
                        description=f"  [dim cyan]Dual-Ollama:[/dim cyan] [white]{model_info}[/white]"
                    )

                summary = run_dual_ollama_consensus(ctx, progress_cb=update_sub_task)
                priority = calculate_priority_score(git_info)

                # Embedding
                progress.update(sub_task, description=f"  [dim green]Computing 768-dim embedding...[/dim green]")
                vector_text = f"Project: {ctx['project_name']}. {summary} Stack: {', '.join(ctx['tech_stack'])}"
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
                    "consensus_summary": summary,
                    "key_files": ctx["top_files"],
                    "embedding_dim": len(embedding),
                    "last_scanned": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "chunks": [
                        {"id": f"{ctx['project_name']}:overview", "text": summary, "type": "consensus_summary"},
                        {"id": f"{ctx['project_name']}:readme", "text": ctx["readme_preview"][:800], "type": "readme_chunk"}
                    ]
                }
                all_projects.append(proj_data)
                all_embeddings.append(embedding)

                progress.advance(total_task)

            progress.update(sub_task, visible=False)
    else:
        for idx, repo_path in enumerate(candidate_paths, 1):
            repo_name = os.path.basename(repo_path)
            print(f"\n[{idx}/{len(candidate_paths)}] Scanning {repo_name}...")
            git_info = inspect_git_status(repo_path)
            ctx = extract_project_context(repo_path)
            summary = run_dual_ollama_consensus(ctx)
            priority = calculate_priority_score(git_info)
            vector_text = f"Project: {ctx['project_name']}. {summary} Stack: {', '.join(ctx['tech_stack'])}"
            embedding = call_ollama_embed(vector_text)
            all_projects.append({
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
                "consensus_summary": summary,
                "key_files": ctx["top_files"],
                "embedding_dim": len(embedding),
                "last_scanned": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "chunks": [
                    {"id": f"{ctx['project_name']}:overview", "text": summary, "type": "consensus_summary"}
                ]
            })
            all_embeddings.append(embedding)

    # Save to local cache
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "projects_cache.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"projects": all_projects}, f, indent=2)
    log_success(f"Saved local cache to [underline]{cache_path}[/underline]")

    # Pinecone Upsert
    if RICH_AVAILABLE:
        with console.status("[bold cyan]Upserting vector embeddings to Pinecone DB...", spinner="aesthetic"):
            pinecone_ok = upsert_to_pinecone(all_projects, all_embeddings)
    else:
        pinecone_ok = upsert_to_pinecone(all_projects, all_embeddings)

    if pinecone_ok:
        log_success(f"Successfully upserted [bold green]{len(all_projects)}[/bold green] vectors to Pinecone Index ([cyan]{PINECONE_INDEX_NAME}[/cyan])!")
    else:
        log_warn("Pinecone DB upsert completed in offline/cache mode.")

    # Vercel Sync
    sync_to_vercel_mcp(all_projects)

    # Render Final Rich Dashboard Table
    console.print()
    render_summary_table(all_projects, pinecone_ok)

    # Final Stats Panel
    synced_count = sum(1 for p in all_projects if p["is_synced"])
    dirty_count = len(all_projects) - synced_count
    if RICH_AVAILABLE:
        stats_text = (
            f"[bold]Total Repositories:[/bold] {len(all_projects)} | "
            f"[bold green]Fully Synced:[/bold green] {synced_count} | "
            f"[bold yellow]Requiring Attention:[/bold yellow] {dirty_count} | "
            f"[bold cyan]Pinecone Vectors:[/bold cyan] {len(all_projects)}"
        )
        console.print(Panel(stats_text, title="📊 Ingestion Summary", border_style="green", expand=False))

if __name__ == "__main__":
    main()
