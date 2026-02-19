# Adjoint Server

[![MCP](https://badge.mcpx.dev?status=on 'MCP Enabled')](https://modelcontextprotocol.io/introduction)
[![python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-red.svg 'MIT License')](https://opensource.org/licenses/MIT)
[![Discord](https://img.shields.io/badge/discord-join-red.svg?logo=discord&logoColor=white)](https://discord.gg/y4p8KfzrN4)

Model Context Protocol server for Unity Editor integration. Control Unity through natural language using AI assistants like Claude, Cursor, and more.

**Adjoint** - AI Development Environment for Unity. This project is not affiliated with Unity Technologies.

**Join our community:** [Discord Server](https://discord.gg/y4p8KfzrN4)

**Required:** Install the Adjoint Unity package to connect Unity Editor with this server.

---

## Installation

### Option 1: Using uvx (Recommended)

Run directly with uvx:

```bash
# HTTP (default)
uvx adjoint-server --transport http --http-url http://localhost:8080

# Stdio
uvx adjoint-server --transport stdio
```

**MCP Client Configuration (HTTP):**

```json
{
  "mcpServers": {
    "UnityMCP": {
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

**MCP Client Configuration (stdio):**

```json
{
  "mcpServers": {
    "UnityMCP": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/adjoint/adjoint-beta-dist#subdirectory=Server",
        "adjoint-server",
        "--transport",
        "stdio"
      ],
      "type": "stdio"
    }
  }
}
```

### Option 2: Using uv (Local Installation)

For local development or custom installations:

```bash
# Clone the repository
git clone https://github.com/adjoint/adjoint-beta-dist.git
cd adjoint-beta-dist/Server

# Run with uv (HTTP)
uv run server.py --transport http --http-url http://localhost:8080

# Run with uv (stdio)
uv run server.py --transport stdio
```

**MCP Client Configuration (HTTP):**
```json
{
  "mcpServers": {
    "UnityMCP": {
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

**MCP Client Configuration (stdio – Windows):**
```json
{
  "mcpServers": {
    "UnityMCP": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "C:\\path\\to\\adjoint-beta-dist\\Server",
        "server.py",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

**MCP Client Configuration (stdio – macOS/Linux):**
```json
{
  "mcpServers": {
    "UnityMCP": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/adjoint-beta-dist/Server",
        "server.py",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

### Option 3: Using Docker

```bash
docker build -t adjoint-server .
docker run -p 8080:8080 adjoint-server --transport http --http-url http://0.0.0.0:8080
```

Configure your MCP client with `"url": "http://localhost:8080/mcp"`. For stdio-in-docker (rare), run the container with `--transport stdio` and use the same `command`/`args` pattern as the uv examples, wrapping it in `docker run -i ...` if needed.

---

## Configuration

The server connects to Unity Editor automatically when both are running. No additional configuration needed.

**Environment Variables:**

- `DISABLE_TELEMETRY=true` - Opt out of anonymous usage analytics
- `LOG_LEVEL=DEBUG` - Enable detailed logging (default: INFO)

---

## Example Prompts

Once connected, try these commands in your AI assistant:

- "Create a 3D player controller with WASD movement"
- "Add a rotating cube to the scene with a red material"
- "Create a simple platformer level with obstacles"
- "Generate a shader that creates a holographic effect"
- "List all GameObjects in the current scene"

---

## Documentation

For complete documentation, troubleshooting, and advanced usage:

**[Full Documentation](https://github.com/adjoint/adjoint-beta-dist#readme)**

---

## Requirements

- **Python:** 3.10 or newer
- **Unity Editor:** 2021.3 LTS or newer
- **uv:** Python package manager ([Installation Guide](https://docs.astral.sh/uv/getting-started/installation/))

---

## License

MIT License - See [LICENSE](https://github.com/adjoint/adjoint-beta-dist/blob/main/LICENSE)
