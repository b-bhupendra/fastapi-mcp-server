"""
Pinecone Store Module.
Encapsulates vector database interactions: Serverless index provisioning,
rich metadata upserts, and cosine similarity queries.
"""

import os
import time
from typing import Dict, List, Any, Optional

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "project-intelligence")

def get_pinecone_client():
    if not PINECONE_API_KEY:
        return None
    try:
        from pinecone import Pinecone
        return Pinecone(api_key=PINECONE_API_KEY)
    except Exception as e:
        print(f"Pinecone client init error: {e}")
        return None

def ensure_index_ready(pc, index_name: str = PINECONE_INDEX_NAME, dimension: int = 768) -> Any:
    indexes = [i.name for i in pc.list_indexes()]
    if index_name not in indexes:
        from pinecone import ServerlessSpec
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(2)
    return pc.Index(index_name)

def upsert_projects(projects: List[Dict[str, Any]], embeddings: List[List[float]]) -> bool:
    pc = get_pinecone_client()
    if not pc:
        return False
    try:
        index = ensure_index_ready(pc)
        vectors_to_upsert = []

        for p, emb in zip(projects, embeddings):
            # 1. Project Overview & Objective Vector
            vec_id = f"proj:{p['project_name']}:overview"
            metadata = {
                "project_name": p.get("project_name", ""),
                "primary_language": p.get("primary_language", ""),
                "intended_objective": p.get("intended_objective", "")[:1000],
                "project_status": p.get("project_status", "In Development"),
                "is_synced": p.get("is_synced", True),
                "unpushed_commits": p.get("unpushed_commits", 0),
                "dirty_files_count": p.get("dirty_files_count", 0),
                "priority_score": p.get("priority_score", 50),
                "consensus_summary": p.get("consensus_summary", "")[:1000],
                "tech_stack": p.get("tech_stack", []),
                "git_remote": p.get("git_remote", ""),
                "git_branch": p.get("git_branch", "main")
            }
            vectors_to_upsert.append({
                "id": vec_id,
                "values": emb,
                "metadata": metadata
            })

        index.upsert(vectors=vectors_to_upsert)
        return True
    except Exception as e:
        print(f"Pinecone upsert failure: {e}")
        return False

def query_vectors(query_embedding: List[float], top_k: int = 5, filter_meta: Optional[Dict[str, Any]] = None):
    pc = get_pinecone_client()
    if not pc:
        return []
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        res = index.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filter_meta,
            include_metadata=True
        )
        return res.get("matches", [])
    except Exception as e:
        print(f"Pinecone query failure: {e}")
        return []
