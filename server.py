# server.py (repo root)
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import the FastMCP app object exported by the repo
from openrouter_mcp.server import mcp  # <-- if your repo uses a different name, change to `app` or `server`
