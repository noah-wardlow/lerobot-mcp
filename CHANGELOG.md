# Changelog

## Unreleased

- Add a `Dockerfile` and `.dockerignore` so the MCP server can run over stdio in a container and be
  auto-inspected by MCP clients and registries.
- Add a `glama.json` manifest and document the Docker run path in the README.
- Add a description to every tool parameter (100% input-schema description coverage) so clients and
  agents get syntax, constraints, and defaults for each argument.
- Add behavioral annotations (title, read-only / destructive / idempotent / open-world hints) to all
  tools, and expand tool docstrings with usage guidance on when to use each one.

## 0.1.4

- Trim redundant preview and generic helper tools from the public MCP surface.

## 0.1.3

- Add tools for agents to find existing LeRobot checkouts and select one for the current MCP session.

## 0.1.2

- Prefer existing LeRobot checkouts in the README, with managed setup documented as a fallback.
- Lazily prepare the managed LeRobot fallback the first time a LeRobot-backed tool needs it.
- Simplify dataset conversion docs so implementation details stay behind the MCP tool surface.

## 0.1.1

- Prepare managed LeRobot checkouts with Python 3.12 by default.
- Install LeRobot's `dataset` extra during managed setup so dataset metadata, conversion, and command
  help work after one MCP install call.
- Expose managed Python and default LeRobot extras in server config, with environment-variable and
  tool-argument overrides.

## 0.1.0

- Initial typed MCP server for LeRobot workflows.
- Dynamic LeRobot command and capability discovery.
- LeRobot command/example dry-run and execution tools.
- Managed background jobs with status, logs, listing, and cancellation.
- Dataset metadata inspection.
- Hugging Face Hub repo and dataset search tools.
- Dataset inspect and convert integration.
