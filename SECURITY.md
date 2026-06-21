# Security

This MCP server runs local LeRobot, Forge, and Hugging Face tooling on behalf of an MCP client.

## Reporting Issues

Please report security issues privately to the repository owner. Do not open a public issue for a
vulnerability until there is a coordinated fix.

## Runtime Safety Model

- The server does not expose arbitrary shell execution.
- LeRobot commands must be discovered from the configured checkout or installed distribution.
- Example scripts must resolve inside the configured LeRobot `examples/` directory.
- Forge commands are built from typed arguments and use a pinned git commit.
- Hardware workflows can move real robots. MCP clients should confirm robot identity, ports,
  workspaces, and operator intent before running record, replay, teleoperate, calibration, motor
  setup, or joint-limit workflows.

Use a dedicated environment when connecting an MCP client to real hardware.
