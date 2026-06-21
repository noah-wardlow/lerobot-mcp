# Contributing

## Development Setup

```bash
uv sync --extra dev
```

Run the standard checks before opening a pull request:

```bash
uv run ruff check .
uv run mypy
uv run pytest -vv
```

For compatibility testing against LeRobot `main`, set `LEROBOT_ROOT` to a local LeRobot checkout:

```bash
LEROBOT_ROOT=/path/to/lerobot uv run lerobot-mcp
```

## Compatibility Expectations

- Prefer dynamic discovery from LeRobot metadata and source files over hardcoded command lists.
- Keep tool inputs structured and typed with Pydantic models.
- Avoid importing heavy robotics dependencies at MCP startup.
- Use foreground commands for short safe checks and background jobs for training, conversion, and hardware workflows.
- Do not add general shell execution tools.

## Forge Pin

Forge integration is intentionally pinned to a specific `main` commit. Update `FORGE_COMMIT` in
`src/lerobot_mcp/config.py` only after testing inspect and dry-run conversion against the new commit.
