# Changelog

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
- Forge inspect and convert integration pinned to commit `461a0179115c7f2dc763ff4b1a1d2de02f5a1e69`.
