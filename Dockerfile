# syntax=docker/dockerfile:1

# Container image for lerobot-mcp.
#
# This builds and runs the MCP server over stdio so MCP clients (and registries
# such as Glama) can launch and introspect the available tools without a local
# Python toolchain. LeRobot-backed tools lazily prepare a managed checkout at
# runtime, so the base image stays small and startup avoids heavy imports.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first using only the lockfile and project metadata so the
# layer caches across source-only changes.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Install the project itself.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Run the MCP server over stdio.
ENTRYPOINT ["lerobot-mcp"]
