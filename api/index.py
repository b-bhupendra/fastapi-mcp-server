import json
from typing import Any
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

app = FastAPI(title="Hello World MCP Server", version="1.0.0")

TOOLS = [
    {
        "name": "hello_world",
        "description": "Say hello to someone or world with a greeting message",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the person to greet",
                    "default": "World"
                }
            }
        }
    },
    {
        "name": "get_server_time",
        "description": "Returns current server status and timestamp",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }
]

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Hello World MCP Server is running! Connect LLM clients to /mcp or /api/mcp",
        "endpoints": {
            "mcp": "/mcp",
            "api_mcp": "/api/mcp",
            "health": "/health"
        },
        "available_tools": [t["name"] for t in TOOLS]
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

async def handle_mcp_request(request: Request) -> Response:
    if request.method == "GET":
        # Returns server description and tools
        return JSONResponse({
            "name": "hello-world-mcp",
            "version": "1.0.0",
            "protocolVersion": "2024-11-05",
            "tools": TOOLS
        })

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None},
            status_code=400
        )

    # If batch request or single JSON-RPC request
    req_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {})

    if method == "initialize":
        response = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {
                        "listChanged": False
                    }
                },
                "serverInfo": {
                    "name": "hello-world-mcp",
                    "version": "1.0.0"
                }
            }
        }
        return JSONResponse(response)

    elif method == "notifications/initialized":
        # Notification - no response required per JSON-RPC, or empty 204
        return Response(status_code=204)

    elif method == "ping":
        return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": {}})

    elif method == "tools/list":
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": TOOLS
            }
        })

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name == "hello_world":
            target = arguments.get("name", "World")
            result_text = f"Hello, {target}! Welcome to your Vercel-hosted FastAPI MCP Server! 🚀"
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": result_text
                        }
                    ]
                }
            })

        elif tool_name == "get_server_time":
            import datetime
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Current server UTC time is: {now}. MCP server running on Vercel serverless."
                        }
                    ]
                }
            })

        else:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Tool '{tool_name}' not found"
                }
            })

    else:
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Method '{method}' not implemented"
            }
        })

@app.api_route("/mcp", methods=["GET", "POST"])
async def mcp_endpoint(request: Request):
    return await handle_mcp_request(request)

@app.api_route("/api/mcp", methods=["GET", "POST"])
async def api_mcp_endpoint(request: Request):
    return await handle_mcp_request(request)
