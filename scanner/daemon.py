#!/usr/bin/env python3
"""
Background Scraper & Ingestion Daemon.
Monitors local repositories, detects changes, runs Dual-Ollama consensus,
and automatically updates Pinecone DB vectors.
"""

import os
import sys
import time
import json
import signal
import datetime
import traceback
from pathlib import Path
from typing import Dict, List, Any

# Ensure scanner package is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from scanner.scanner import (
    scan_repository,
    upsert_to_pinecone,
    sync_to_vercel_mcp,
    inspect_git_status,
    log_info,
    log_success,
    log_warn,
    PINECONE_INDEX_NAME
)

WATCH_INTERVAL = int(os.environ.get("SCAN_INTERVAL_SECONDS", "300"))
TARGET_DIRS = [os.environ.get("SCAN_TARGET_DIR", "/home/bhupendra/.gemini/antigravity-ide/scratch")]
STATE_FILE = os.path.join(root_dir, "logs", "daemon_state.json")

RUNNING = True

def signal_handler(signum, frame):
    global RUNNING
    print(f"\n[!] Received signal {signum}. Gracefully shutting down daemon...", flush=True)
    RUNNING = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def load_state() -> Dict[str, str]:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_state(state: Dict[str, str]):
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass

def get_repo_state_hash(repo_path: str) -> str:
    """Computes quick fingerprint of repo to detect if rescanning is needed."""
    try:
        git_info = inspect_git_status(repo_path)
        return f"{git_info.get('git_branch')}:{git_info.get('unpushed_commits')}:{git_info.get('dirty_files_count')}:{git_info.get('last_commit_date')}"
    except Exception:
        return ""

def run_daemon_cycle():
    state = load_state()
    candidate_paths = []
    for base_dir in TARGET_DIRS:
        if not os.path.exists(base_dir):
            continue
        if os.path.isdir(os.path.join(base_dir, ".git")):
            candidate_paths.append(base_dir)
            continue
        for item in sorted(os.listdir(base_dir)):
            full_path = os.path.join(base_dir, item)
            if os.path.isdir(full_path) and not item.startswith("."):
                candidate_paths.append(full_path)

    updated_projects = []
    updated_embeddings = []

    for path in candidate_paths:
        name = os.path.basename(path)
        current_hash = get_repo_state_hash(path)
        last_hash = state.get(path)

        if current_hash != last_hash:
            log_info(f"🔄 Changes detected in: [bold]{name}[/bold]. Running Dual-Ollama consensus...")
            try:
                proj, emb = scan_repository(path)
                updated_projects.append(proj)
                updated_embeddings.append(emb)
                state[path] = current_hash
                save_state(state)
            except Exception as e:
                log_warn(f"Failed scanning {name}: {e}")

    if updated_projects:
        log_info(f"Upserting {len(updated_projects)} updated project vectors to Pinecone DB...")
        try:
            upsert_to_pinecone(updated_projects, updated_embeddings)
            sync_to_vercel_mcp(updated_projects)
            log_success(f"Successfully synchronized {len(updated_projects)} projects to Pinecone & Vercel!")
        except Exception as e:
            log_warn(f"Sync error: {e}")
    else:
        log_info(f"✔ All {len(candidate_paths)} repositories are in sync with Pinecone DB.")

def main():
    log_info(f"🚀 Project Intelligence Background Daemon active (Interval: {WATCH_INTERVAL}s)")
    log_info(f"Target Directory: {', '.join(TARGET_DIRS)}")
    
    while RUNNING:
        try:
            run_daemon_cycle()
        except Exception as e:
            log_warn(f"Daemon cycle exception: {e}")
            traceback.print_exc()

        for _ in range(WATCH_INTERVAL):
            if not RUNNING:
                break
            time.sleep(1)

    log_info("Daemon stopped gracefully.")

if __name__ == "__main__":
    main()
