# server.py  (repo root)
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Bind the FastMCP app object exported by the package
from openrouter_mcp.server import mcp as mcp

# Force-load handlers so their @mcp.tool decorators register
from openrouter_mcp.handlers import chat, multimodal, mcp_benchmark, collective_intelligence

# (Optional sanity tool)
@mcp.tool
async def echo(text: str):
    return {"content":[{"type":"text","text":text}]}
