## Quick context for AI coding agents

This repo implements an MCP server (Python) backed by OpenRouter plus a thin Node CLI. The Python MCP server in `src/openrouter_mcp/` is the primary development surface; the CLI in `bin/` is thin glue for init/start flows.

Key points to be immediately productive:

- Entrypoint & app object: `src/openrouter_mcp/server.py` creates `mcp = FastMCP("openrouter-mcp")`. Always import that `mcp` when registering tools: `from src.openrouter_mcp.server import mcp`.
- Handlers: add new tools under `src/openrouter_mcp/handlers/`. Register functions with `@mcp.tool()` and keep modules side-effect free (no long-running work at import).
- Environment: `OPENROUTER_API_KEY` is required. Validation lives in `create_app()` / `validate_environment()` in `server.py` and will raise if missing.
- Multimodal: follow `src/openrouter_mcp/handlers/multimodal.py` for image input handling (path/url/base64), encoding, resizing, and tool shapes (`chat_with_vision`, `list_vision_models`).

Developer workflows (concrete commands):

- Run tests: `python -m pytest tests/ -v` (also via `npm test`).
- Lint: `python -m ruff check src/ tests/`.
- Format: `python -m black src/ tests/`.
- Start server (dev): ensure `.env` or env var `OPENROUTER_API_KEY` is set, then run `python -m src.openrouter_mcp.server` (or `npx @physics91/openrouter-mcp start` which calls the CLI).

Conventions and gotchas specific to this project:

- Single shared FastMCP instance: do not instantiate FastMCP elsewhere; all tools must register on the exported `mcp` from `server.py`.
- Tests import `src.openrouter_mcp.server.mcp` directly; adding/removing tools may affect test discovery—use existing tests as canonical examples.
- If you add Python dependencies, update `requirements.txt` and `requirements-dev.txt`. Node-side changes belong only in `bin/` and `package.json`.

Integration & references:

- Node CLI: `bin/openrouter-mcp.js` — used for `init` (creates `.env`) and `start` flows. See `package.json` scripts for wrappers.
- Tool contracts: `docs/API.md` (use this for exact request/response shapes).
- Client helpers: `src/openrouter_mcp/client/` (OpenRouter client wrappers used by handlers).

Where to look for quick examples:

- App creation & env checks: `src/openrouter_mcp/server.py`
- Multimodal handling: `src/openrouter_mcp/handlers/multimodal.py`
- Tool registration patterns: files under `src/openrouter_mcp/handlers/` (e.g., `chat.py`, `mcp_benchmark.py`)
- Tests demonstrating tool calls: `tests/test_mcp_server.py`, `tests/test_multimodal.py`

If you'd like, I can add a handler template + matching unit test (one-file PR) to demonstrate the canonical pattern — tell me where you'd prefer it placed.
## Quick context for AI coding agents

This repository implements the OpenRouter Model Context Protocol (MCP) server (Python) plus a Node.js CLI. Focus on the Python MCP server in `src/openrouter_mcp/` and the CLI glue in `bin/`.

High-level architecture:
- FastMCP-based Python server exporting `mcp` in `src/openrouter_mcp/server.py` (entrypoint used by `server.py`). Handlers live in `src/openrouter_mcp/handlers/` and register tools with `@mcp.tool`.
- Node CLI (`bin/openrouter-mcp.js`) wraps npm commands and forwards to the Python backend for local development and installation flows.
- Docs live under `docs/` (API.md contains tool contracts and examples) and README.md contains CLI quick-start.

Key developer workflows and commands:
- Run tests: `python -m pytest tests/ -v` (package.json `test` script)
- Run linter/format: `python -m ruff check src/ tests/` and `python -m black src/ tests/` (see `package.json` scripts)
- Start server for local development: set `OPENROUTER_API_KEY` (or run `npx @physics91/openrouter-mcp init`), then run `python -m src.openrouter_mcp.server` or `npx @physics91/openrouter-mcp start` which invokes the CLI wrapper.

Project-specific patterns and conventions to follow:
- The FastMCP app object is created in `src/openrouter_mcp/server.py` as `mcp = FastMCP('openrouter-mcp')`. New tools must import that `mcp` and use `@mcp.tool` to register.
- Handlers should be side-effect free at import time except for decorator registration. Avoid long-running work at import.
- Environment validation is centralized in `src/openrouter_mcp/server.py`. Required env var: `OPENROUTER_API_KEY`.
- Images in multimodal handlers accept file paths, URLs, or base64 strings—follow existing parsing helpers in `src/openrouter_mcp/handlers/multimodal.py`.

Integration and external dependencies:
- OpenRouter API (requires `OPENROUTER_API_KEY`) — many components call out to provider-specific endpoints via the unified OpenRouter model listing.
- FastMCP is the runtime framework — tools are async and may stream via MCP streaming chunks.
- Node.js CLI is used to bootstrap configs and run helper installers (claude integration). Tests assume Python 3.9+.

Testing and examples:
- The `tests/` folder contains unit and integration-style tests that import `src.openrouter_mcp.server.mcp`. Use these as canonical examples for calling tools and expected tool names (e.g., `benchmark_models`, `chat_with_model`).
- Use `docs/API.md` for exact tool parameter/response shapes when generating or modifying handlers.

Editing guidance for AI agents:
- Prefer small, focused changes. When adding a new tool handler, create a single module under `src/openrouter_mcp/handlers/`, register with `@mcp.tool`, add docs in `docs/` and unit tests in `tests/`.
- Keep CLI changes isolated under `bin/` and Node-related packaging in `package.json`.
- Run tests and lint after edits. If adding dependencies, update `requirements.txt` and `requirements-dev.txt`.

Examples (use these exact file references when generating code):
- create app: `from src.openrouter_mcp.server import mcp`
- start dev server: `python -m src.openrouter_mcp.server`
- tests: `python -m pytest tests/ -v`

If anything is ambiguous or you need runtime secrets (API keys), ask the developer before making changes.
