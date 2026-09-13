# Project Intelligence MCP Server (Central Pinecone DB + Dual-Ollama Consensus)

A serverless-ready **Model Context Protocol (MCP)** platform built with **FastAPI**, backed by **Pinecone DB** as the central cloud knowledge & vector repository, designed for instant deployment on **Vercel** and seamless RAG analysis with **Gemini Spark**, Cursor, and Claude.

---

## 🏗️ Architecture Overview

```
 ┌────────────────────────────────────────────────────────────┐
 │ Local Workstation                                          │
 │  • Local Directory Scanner (~/Desktop, ~/projects)         │
 │  • Git Status Inspector (dirty files, unpushed commits)    │
 │  • Context Extractor (README, manifests, code entry)       │
 │                                                            │
 │  🤖 Dual-Ollama Multi-Temperature Consensus Engine        │
 │     - qwen2.5:7b   (T = 0.0, 0.5, 0.9)                     │
 │     - llama3.2:7b  (T = 0.0, 0.5, 0.9)                     │
 │     - Consensus Voter & Hallucination Filter               │
 │                                                            │
 │  • Embeddings Generator (nomic-embed-text, 768-dim)       │
 └─────────────────────────────┬──────────────────────────────┘
                               │ Upsert Vectors & Metadata
                               ▼
 ┌────────────────────────────────────────────────────────────┐
 │ Central Repository: Pinecone DB (Cloud Vector Store)       │
 │  • Dense Code & Doc Embeddings                             │
 │  • Rich Metadata (git status, priority, tech stack, branch)│
 └─────────────────────────────┬──────────────────────────────┘
                               │ Query & Metadata Filter
                               ▼
 ┌────────────────────────────────────────────────────────────┐
 │ Vercel Serverless MCP Platform                             │
 │  • FastAPI Backend (/mcp JSON-RPC 2.0 endpoint)            │
 │  • Zero-state, scale-to-zero, instant response             │
 └─────────────────────────────┬──────────────────────────────┘
                               │ MCP Protocol
                               ▼
 ┌────────────────────────────────────────────────────────────┐
 │ AI Clients (Gemini Spark, Cursor, Claude Desktop)          │
 │  • Semantic Code & Project RAG                             │
 │  • Project Prioritization & Git Sync Alerting              │
 │  • Deep Architectural Audits                               │
 └────────────────────────────────────────────────────────────┘
```

---

## 🛠️ MCP Tools for Gemini Spark & LLMs

| Tool Name | Description | Key Parameters |
| :--- | :--- | :--- |
| `query_projects_rag` | **Semantic RAG Search**: Queries Pinecone DB for relevant code snippets, architecture docs, and summaries matching natural language queries. | `query` (str), `top_k` (int), `filter_synced` (bool) |
| `list_projects` | **Project Inventory**: Lists tracked repositories with primary languages, branches, sync states, and priority scores. | `dirty_only` (bool), `language` (str), `limit` (int) |
| `get_project_context` | **Project Dossier**: Retrieves verified Dual-Ollama consensus summary, tech stack, and key files for a specific project. | `project_name` (str) |
| `check_git_sync` | **Sync Inspector**: Identifies repos with uncommitted files or unpushed commits needing backup to GitHub. | `project_name` (optional str) |
| `prioritize_projects` | **Urgency Ranking**: Ranks projects by calculated urgency (unpushed work, recent changes). | `top_n` (int) |
| `analyze_project_deep_dive` | **Deep Multi-Chunk Analysis**: Delivers multi-chunk context tailored for Gemini Spark to analyze architecture, dependencies, or recommend next steps. | `project_name` (str), `analysis_focus` (str) |

---

## 🚀 Quick Start

### 1. Local Scanner & Dual-Ollama Consensus
Run the local scanner to evaluate your local repositories, run consensus across `qwen2.5:7b` & `llama3.2:latest`, and upsert vectors to Pinecone:

```bash
# Optional: Set your Pinecone credentials
export PINECONE_API_KEY="your-pinecone-api-key"
export PINECONE_INDEX_NAME="project-intelligence"

# Run scanner across your project directories
python scanner/scanner.py /path/to/your/projects
```

### 2. Run Local MCP Server
```bash
uvicorn api.index:app --reload --port 8000
```

### 3. Deploy to Vercel
1. Push changes to GitHub:
   ```bash
   git add .
   git commit -m "feat: Project Intelligence MCP with Pinecone DB & Dual-Ollama consensus"
   git push origin main
   ```
2. In Vercel Project Settings > Environment Variables:
   - `PINECONE_API_KEY`: Your Pinecone API key
   - `PINECONE_INDEX_NAME`: `project-intelligence` (or your chosen index name)
   - `SYNC_SECRET`: Secret token for scanner synchronization

---

## 🤖 Connecting to Gemini Spark / Cursor / Claude
- **Server URL**: `https://fastapi-mcp-server.vercel.app/mcp`
- **Protocol**: HTTP JSON-RPC 2.0
