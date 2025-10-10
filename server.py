# server.py (repo root)
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# 1) Import the FastMCP app object exported by the repo
#    If the repo’s object is named something else (e.g., `app`), just alias it to mcp.
from openrouter_mcp.server import mcp as mcp

# 2) (Important) Import the tool modules so decorators run and tools register.
#    Search the repo for '@mcp.tool' and import those modules here.
#    Examples (adjust to match your fork):
from openrouter_mcp.tools import collective, chat, models
from openrouter_mcp.tools import consensus, ensemble

# If you’re unsure of the exact module names, add a trivial tool temporarily:
from fastmcp import FastMCP
@mcp.tool
async def echo(text: str): return {"content":[{"type":"text","text":text}]}
