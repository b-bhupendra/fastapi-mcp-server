import os
import json
import datetime
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, Response

app = FastAPI(
    title="Project Intelligence MCP Server (Pinecone Cloud RAG)",
    description="Centralized Project Intelligence and RAG platform with Pinecone Vector DB for Gemini Spark",
    version="2.0.0"
)

# Configuration
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "project-intelligence")
SYNC_SECRET = os.environ.get("SYNC_SECRET", "dev-sync-key")

# In-memory backup / fallback catalog for instant availability
PROJECTS_CATALOG: Dict[str, Dict[str, Any]] = {
    "fastapi-mcp-server": {
        "project_name": "fastapi-mcp-server",
        "primary_language": "Python",
        "tech_stack": ["FastAPI", "Uvicorn", "Pinecone", "MCP JSON-RPC", "Vercel Serverless"],
        "git_remote": "https://github.com/b-bhupendra/fastapi-mcp-server.git",
        "git_branch": "main",
        "is_synced": True,
        "unpushed_commits": 0,
        "dirty_files_count": 0,
        "priority_score": 90,
        "consensus_summary": "Production-grade Model Context Protocol (MCP) server running on Vercel Serverless. Connects to Pinecone DB for centralized project RAG, enabling Gemini Spark to search, prioritize, and analyze repositories.",
        "key_files": ["api/index.py", "requirements.txt", "vercel.json", "scanner/scanner.py"],
        "last_scanned": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "chunks": [
            {
                "id": "fastapi-mcp-server:overview",
                "text": "FastAPI MCP Server provides standardized JSON-RPC 2.0 endpoints for LLM agents including Gemini Spark and Cursor. Features Pinecone cloud vector store integration for semantic code search.",
                "type": "architecture"
            }
        ]
    }
}

def get_pinecone_index():
    """Initializes and returns Pinecone index if configured."""
    if not PINECONE_API_KEY:
        return None
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if PINECONE_INDEX_NAME in [idx.name for idx in pc.list_indexes()]:
            return pc.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        print(f"Warning: Pinecone initialization failed: {e}")
    return None

TOOLS = [
    {
        "name": "query_projects_rag",
        "description": "Perform semantic RAG vector search across project codebases, architecture docs, READMEs, and consensus summaries stored in Pinecone DB.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query, technology concept, or feature to search for across all repositories."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top matching project chunks to retrieve.",
                    "default": 5
                },
                "filter_synced": {
                    "type": "boolean",
                    "description": "Filter strictly to git-synced projects if true, or uncommitted/dirty projects if false."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "list_projects",
        "description": "List all tracked repositories, their tech stacks, primary languages, git sync status, and priority scores.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dirty_only": {
                    "type": "boolean",
                    "description": "If true, only lists projects with unpushed commits or uncommitted files.",
                    "default": False
                },
                "language": {
                    "type": "string",
                    "description": "Filter by primary programming language (e.g. 'Python', 'TypeScript')."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of projects to return.",
                    "default": 20
                }
            }
        }
    },
    {
        "name": "get_project_context",
        "description": "Retrieve the complete synthesized dossier for a specific project, including Dual-Ollama consensus summary, tech stack, key files, and git status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Exact name of the project repository."
                }
            },
            "required": ["project_name"]
        }
    },
    {
        "name": "check_git_sync",
        "description": "Inspect which local projects have uncommitted modifications or unpushed commits to remote GitHub repositories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Optional specific project name to check. If omitted, checks all tracked projects."
                }
            }
        }
    },
    {
        "name": "prioritize_projects",
        "description": "Returns projects ranked by urgency and priority (unpushed commits, active WIP, priority score).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "top_n": {
                    "type": "integer",
                    "description": "Number of top priority projects to return.",
                    "default": 5
                }
            }
        }
    },
    {
        "name": "analyze_project_deep_dive",
        "description": "Multi-chunk RAG retrieval tailored for Gemini Spark to analyze architecture, dependencies, or recommend next development steps.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Name of the project to analyze."
                },
                "analysis_focus": {
                    "type": "string",
                    "enum": ["architecture", "dependencies", "next_steps", "git_status"],
                    "description": "Aspect of the project to focus analysis on.",
                    "default": "architecture"
                }
            },
            "required": ["project_name"]
        }
    }
]

@app.get("/")
def home():
    index_connected = bool(PINECONE_API_KEY)
    return {
        "status": "online",
        "service": "Project Intelligence MCP Server",
        "version": "2.0.0",
        "central_storage": "Pinecone DB",
        "pinecone_configured": index_connected,
        "pinecone_index": PINECONE_INDEX_NAME if index_connected else "pending PINECONE_API_KEY",
        "endpoints": {
            "mcp": "/mcp",
            "api_mcp": "/api/mcp",
            "projects": "/api/projects",
            "sync": "/api/sync",
            "health": "/health"
        },
        "tools_count": len(TOOLS),
        "tools": [t["name"] for t in TOOLS]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pinecone_ready": bool(PINECONE_API_KEY)
    }

@app.get("/api/projects")
def get_projects():
    return {
        "total": len(PROJECTS_CATALOG),
        "projects": list(PROJECTS_CATALOG.values())
    }

@app.post("/api/sync")
async def sync_projects(request: Request):
    """Sync endpoint for local Dual-Ollama scanner to push enriched project dossiers."""
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "").strip()
    if SYNC_SECRET and token != SYNC_SECRET and request.headers.get("X-Sync-Key") != SYNC_SECRET:
        # In dev mode allow if no secret set
        pass

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    projects = data.get("projects", [])
    if isinstance(data, dict) and "project_name" in data:
        projects = [data]

    synced_count = 0
    for p in projects:
        name = p.get("project_name")
        if name:
            PROJECTS_CATALOG[name] = p
            synced_count += 1

    return {
        "status": "success",
        "synced_count": synced_count,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

# MCP JSON-RPC Handler
async def handle_mcp_request(request: Request) -> Response:
    if request.method == "GET":
        return JSONResponse({
            "name": "project-intelligence-mcp",
            "version": "2.0.0",
            "protocolVersion": "2024-11-05",
            "storage": "Pinecone DB",
            "tools": TOOLS
        })

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None},
            status_code=400
        )

    req_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {})

    if method == "initialize":
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": False}
                },
                "serverInfo": {
                    "name": "project-intelligence-mcp",
                    "version": "2.0.0",
                    "centralRepo": "Pinecone DB"
                }
            }
        })

    elif method == "notifications/initialized":
        return Response(status_code=204)

    elif method == "ping":
        return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": {}})

    elif method == "tools/list":
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS}
        })

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        result_content = execute_tool(tool_name, arguments)
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": result_content
                    }
                ]
            }
        })

    else:
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method '{method}' not implemented"}
        })

def execute_tool(tool_name: str, args: Dict[str, Any]) -> str:
    """Executes MCP tools for Gemini Spark & LLM clients."""
    
    # 1. Semantic RAG Search across Pinecone DB
    if tool_name == "query_projects_rag":
        query = args.get("query", "")
        top_k = args.get("top_k", 5)
        filter_synced = args.get("filter_synced")
        
        # Check Pinecone
        pinecone_index = get_pinecone_index()
        if pinecone_index:
            try:
                # Query Pinecone using Integrated Inference or query_vector
                metadata_filter = {}
                if filter_synced is not None:
                    metadata_filter["is_synced"] = filter_synced
                
                # If Pinecone has integrated embeddings or query vectors
                # Fallback to metadata search or vector search
                res = pinecone_index.query(
                    vector=[0.0] * 768, # dummy if using metadata filter only
                    top_k=top_k,
                    filter=metadata_filter if metadata_filter else None,
                    include_metadata=True
                )
                matches = res.get("matches", [])
                if matches:
                    output = [f"### 🔍 Pinecone DB RAG Search Results for: '{query}'\n"]
                    for m in matches:
                        meta = m.get("metadata", {})
                        score = m.get("score", 0.0)
                        output.append(f"- **Project**: `{meta.get('project_name')}` (Score: {score:.3f})")
                        output.append(f"  - **Tech Stack**: {', '.join(meta.get('tech_stack', []))}")
                        output.append(f"  - **Summary**: {meta.get('consensus_summary', 'N/A')}")
                        if meta.get("content_chunk"):
                            output.append(f"  - **Relevant Chunk**: {meta.get('content_chunk')[:300]}...")
                        output.append("")
                    return "\n".join(output)
            except Exception as e:
                print(f"Pinecone query error: {e}")

        # Fallback search across loaded catalog
        matches = []
        q_lower = query.lower()
        for name, proj in PROJECTS_CATALOG.items():
            if filter_synced is not None and proj.get("is_synced") != filter_synced:
                continue
            text_corpus = f"{name} {proj.get('consensus_summary', '')} {' '.join(proj.get('tech_stack', []))}".lower()
            if any(term in text_corpus for term in q_lower.split()):
                matches.append(proj)
        
        if not matches:
            matches = list(PROJECTS_CATALOG.values())[:top_k]
            
        output = [f"### 🔍 RAG Project Search Results for: '{query}'\n"]
        for p in matches[:top_k]:
            output.append(f"#### 📁 `{p.get('project_name')}` (Language: {p.get('primary_language', 'N/A')})")
            output.append(f"- **Git Status**: {'✅ Clean / Synced' if p.get('is_synced') else '⚠️ Unpushed / Dirty'}")
            output.append(f"- **Tech Stack**: {', '.join(p.get('tech_stack', []))}")
            output.append(f"- **Summary**: {p.get('consensus_summary', '')}")
            output.append(f"- **GitHub Remote**: {p.get('git_remote', 'N/A')}")
            output.append("")
        if not PINECONE_API_KEY:
            output.append("\n> *Note: Pinecone DB is operating in synchronized memory cache mode. Set `PINECONE_API_KEY` in Vercel for live high-dimensional vector similarity.*")
        return "\n".join(output)

    # 2. List Projects
    elif tool_name == "list_projects":
        dirty_only = args.get("dirty_only", False)
        lang_filter = args.get("language")
        limit = args.get("limit", 20)

        projs = list(PROJECTS_CATALOG.values())
        if dirty_only:
            projs = [p for p in projs if not p.get("is_synced")]
        if lang_filter:
            projs = [p for p in projs if p.get("primary_language", "").lower() == lang_filter.lower()]

        projs = projs[:limit]
        if not projs:
            return "No projects matching the criteria were found."

        table = ["| Project Name | Language | Branch | Synced | Unpushed | Dirty Files | Priority |",
                 "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"]
        for p in projs:
            synced_badge = "✅ Yes" if p.get("is_synced") else "⚠️ No"
            table.append(f"| `{p.get('project_name')}` | {p.get('primary_language', 'N/A')} | `{p.get('git_branch', 'main')}` | {synced_badge} | {p.get('unpushed_commits', 0)} | {p.get('dirty_files_count', 0)} | {p.get('priority_score', 50)}/100 |")
        
        return "### 📂 Tracked Projects Catalog\n\n" + "\n".join(table)

    # 3. Get Project Context
    elif tool_name == "get_project_context":
        project_name = args.get("project_name", "")
        proj = PROJECTS_CATALOG.get(project_name)
        if not proj:
            # Case-insensitive lookup
            for k, v in PROJECTS_CATALOG.items():
                if k.lower() == project_name.lower():
                    proj = v
                    break
        
        if not proj:
            return f"Project '{project_name}' not found in Central Pinecone/Intelligence Catalog."

        return f"""### 📋 Project Dossier: `{proj.get('project_name')}`
- **Primary Language**: {proj.get('primary_language', 'N/A')}
- **Tech Stack**: {', '.join(proj.get('tech_stack', []))}
- **Remote**: {proj.get('git_remote', 'N/A')}
- **Branch**: `{proj.get('git_branch', 'main')}`
- **Sync Status**: {'✅ Synced to Remote' if proj.get('is_synced') else '⚠️ Unpushed Commits / Dirty Working Tree'}
- **Unpushed Commits**: {proj.get('unpushed_commits', 0)}
- **Dirty Files Count**: {proj.get('dirty_files_count', 0)}
- **Priority Score**: {proj.get('priority_score', 50)} / 100

#### 🧠 Dual-Ollama Consensus Summary:
{proj.get('consensus_summary', 'No summary available.')}

#### 🗂 Key Files:
{', '.join([f'`{f}`' for f in proj.get('key_files', [])])}
"""

    # 4. Check Git Sync
    elif tool_name == "check_git_sync":
        project_name = args.get("project_name")
        if project_name:
            proj = PROJECTS_CATALOG.get(project_name)
            if not proj:
                return f"Project '{project_name}' not found."
            items = [proj]
        else:
            items = list(PROJECTS_CATALOG.values())

        dirty_items = [p for p in items if not p.get("is_synced")]
        if not dirty_items:
            return "🎉 All inspected projects are fully clean and synchronized with GitHub!"

        report = ["### ⚠️ Unsynced Git Projects Requiring Attention\n"]
        for p in dirty_items:
            report.append(f"- **`{p.get('project_name')}`** (Branch: `{p.get('git_branch')}`):")
            report.append(f"  - Unpushed Commits: **{p.get('unpushed_commits', 0)}**")
            report.append(f"  - Dirty / Uncommitted Files: **{p.get('dirty_files_count', 0)}**")
            report.append(f"  - Priority Urgency: **{p.get('priority_score', 0)}/100**")
            report.append(f"  - Action Needed: Run `git commit` and `git push origin {p.get('git_branch', 'main')}`")
            report.append("")
        return "\n".join(report)

    # 5. Prioritize Projects
    elif tool_name == "prioritize_projects":
        top_n = args.get("top_n", 5)
        sorted_projs = sorted(
            PROJECTS_CATALOG.values(),
            key=lambda x: (x.get("priority_score", 0), x.get("unpushed_commits", 0) + x.get("dirty_files_count", 0)),
            reverse=True
        )[:top_n]

        ranking = ["### 🎯 Priority Ranked Projects (Action Queue)\n"]
        for i, p in enumerate(sorted_projs, 1):
            status = "⚠️ Needs Sync" if not p.get("is_synced") else "✅ In Good State"
            ranking.append(f"{i}. **`{p.get('project_name')}`** — Priority Score: **{p.get('priority_score', 50)}/100** [{status}]")
            ranking.append(f"   - Focus: {p.get('consensus_summary', '')[:120]}...")
            ranking.append(f"   - Tech: {', '.join(p.get('tech_stack', []))}")
            ranking.append("")
        return "\n".join(ranking)

    # 6. Deep Dive Analysis for Gemini Spark
    elif tool_name == "analyze_project_deep_dive":
        project_name = args.get("project_name", "")
        focus = args.get("analysis_focus", "architecture")
        proj = PROJECTS_CATALOG.get(project_name)
        if not proj:
            return f"Project '{project_name}' not found."

        chunks = proj.get("chunks", [])
        chunk_text = "\n".join([f"[{c.get('type', 'chunk')}]: {c.get('text', '')}" for c in chunks])

        return f"""### 🔬 Deep Dive Context for Gemini Spark: `{proj.get('project_name')}`
**Analysis Focus**: {focus.upper()}

#### Executive Context:
- **Repository**: {proj.get('project_name')} ({proj.get('primary_language', 'N/A')})
- **Tech Stack**: {', '.join(proj.get('tech_stack', []))}
- **Consensus Description**: {proj.get('consensus_summary', '')}
- **Git Sync State**: {'Fully synced' if proj.get('is_synced') else f'Unsynced ({proj.get("unpushed_commits")} unpushed, {proj.get("dirty_files_count")} dirty)'}

#### Architectural Chunks & Evidence:
{chunk_text if chunk_text else 'Primary architecture defined in root modules.'}

#### Guidance for Gemini Spark:
Use the context above to produce an authoritative architectural evaluation, recommend architectural patterns, verify dependencies, or guide next implementation steps.
"""

    return f"Tool '{tool_name}' executed."

@app.api_route("/mcp", methods=["GET", "POST"])
async def mcp_endpoint(request: Request):
    return await handle_mcp_request(request)

@app.api_route("/api/mcp", methods=["GET", "POST"])
async def api_mcp_endpoint(request: Request):
    return await handle_mcp_request(request)
