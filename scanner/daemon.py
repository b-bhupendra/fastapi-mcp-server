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
from pathlib import Path
from typing import Dict, List, Any

# Ensure scanner package is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

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

LAST_SEEN_STATES: Dict[str, str] = {}
RUNNING = True

def signal_handler(signum, frame):
    global RUNNING
    print("\n[!] Received stop signal. Gracefully stopping daemon...")
    RUNNING = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def get_repo_state_hash(repo_path: str) -> str:
    """Computes quick fingerprint of repo to detect if rescanning is needed."""
    git_info = inspect_git_status(repo_path)
    return f"{git_info['git_branch']}:{git_info['unpushed_commits']}:{git_info['dirty_files_count']}:{git_info['last_commit_date']}"

def run_daemon_cycle():
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
        last_hash = LAST_SEEN_STATES.get(path)

        if current_hash != last_hash:
            log_info(f"🔄 Detected changes in repository: [bold]{name}[/bold]. Running Dual-Ollama consensus...")
            try:
                proj, emb = scan_repository(path)
                updated_projects.append(proj)
                updated_embeddings.append(emb)
                LAST_SEEN_STATES[path] = current_hash
            except Exception as e:
                log_warn(f"Failed scanning {name}: {e}")

    if updated_projects:
        log_info(f"Upserting {len(updated_projects)} updated project vectors to Pinecone DB...")
        upsert_to_pinecone(updated_projects, updated_embeddings)
        sync_to_vercel_mcp(updated_projects)
        log_success(f"Successfully synchronized {len(updated_projects)} projects to Pinecone & Vercel!")
    else:
        log_info(f"All {len(candidate_paths)} repositories are up to date with Pinecone DB.")

def main():
    log_info(f"🚀 Project Intelligence Background Daemon started (Interval: {WATCH_INTERVAL}s)")
    log_info(f"Target Directory: {', '.join(TARGET_DIRS)}")
    
    while RUNNING:
        try:
            run_daemon_cycle()
        except Exception as e:
            log_warn(f"Daemon cycle error: {e}")

        for _ in range(WATCH_INTERVAL):
            if not RUNNING:
                break
            time.sleep(1)

    log_info("Daemon stopped successfully.")

if __name__ == "__main__":
    main()
