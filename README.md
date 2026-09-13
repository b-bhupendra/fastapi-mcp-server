# FastAPI Hello World MCP Server for Vercel

A lightweight, serverless-ready **Model Context Protocol (MCP)** server built with **FastAPI**, designed for instant zero-config deployment on **Vercel** and compatible with MCP-capable LLMs and AI platforms (Gemini Spark / Cursor / Claude Desktop / Antigravity / OpenAI Agents).

---

## 🚀 Features

- **MCP Protocol Compliant**: Supports JSON-RPC 2.0 `initialize`, `tools/list`, and `tools/call`.
- **Pre-configured Tools**:
  1. `hello_world`: Say hello to anyone with a greeting.
  2. `get_server_time`: Returns current UTC time and serverless status.
- **Fast & Serverless**: Zero state, scale-to-zero, instant cold starts.
- **Vercel Native**: Configured with `vercel.json` and standard Python Serverless Function entry point (`api/index.py`).

---

## 🛠️ Local Testing

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run local server:
```bash
uvicorn api.index.py:app --reload --port 8000
```

3. Test MCP initialize:
```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}'
```

4. Test MCP tool call:
```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "hello_world", "arguments": {"name": "Gemini"}}}'
```

---

## 🌐 Deploy to Vercel via GitHub

1. Create a new repository on GitHub (e.g. `fastapi-mcp-server`).
2. Push this directory to your repository:
   ```bash
   git init
   git add .
   git commit -m "feat: hello world mcp server on fastapi"
   git branch -M main
   git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
   git push -u origin main
   ```
3. Go to [vercel.com/new](https://vercel.com/new).
4. Import your GitHub repository.
5. Click **Deploy** (No special environment variables or build commands required).

---

## 🤖 Connecting to Gemini Spark / Cursor / Claude Desktop

Once deployed, your live URL will be:
```
https://<YOUR-VERCEL-PROJECT-NAME>.vercel.app/mcp
```
(or `https://<YOUR-VERCEL-PROJECT-NAME>.vercel.app/api/mcp`)

### In Cursor / Gemini Spark / MCP Config:
- **Type**: `HTTP` / `Streamable HTTP` / `SSE`
- **URL**: `https://<YOUR-PROJECT>.vercel.app/mcp`
