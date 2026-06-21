# lerobot-mcp

MCP server for LeRobot workflows.

`lerobot-mcp` gives MCP clients a structured, auditable interface over the current LeRobot CLI,
examples, source registries, datasets, and dataset conversion workflows.

## Features

- Discover available `lerobot-*` entry points from a managed, local, or installed LeRobot checkout.
- List and run scripts under LeRobot's `examples/` tree with path traversal protection.
- Audit registered policies, rewards, robots, teleoperators, cameras, envs, processors, rollout
  strategies, optimizers, schedulers, and RL algorithms by static source inspection.
- Build dry-run LeRobot commands from structured MCP arguments.
- Run LeRobot commands as foreground calls or managed background jobs.
- Inspect LeRobot dataset metadata without importing heavy robotics dependencies at MCP startup.
- Optionally use Hub auth from your existing environment.
- Convert robotics datasets with Forge pinned to a known bug-fix commit on `main`.
- Search Hugging Face and Forge-registry datasets by robot, format, task, size, episode count, and
  compatibility hints.

## Install

From PyPI:

```bash
uv tool install lerobot-mcp
```

From a checkout:

```bash
git clone https://github.com/noah-wardlow/lerobot-mcp.git
cd lerobot-mcp
uv sync --extra dev
```

## MCP Quick Start

You do not need to clone LeRobot manually for the normal path. Configure the MCP server, then ask your
client to run `lerobot_install_or_update_lerobot`. That tool clones or updates LeRobot `main` into the
managed checkout at `~/.cache/lerobot-mcp/lerobot`, then prepares its local `uv` environment with
Python 3.12 and LeRobot's `dataset` extra. That default covers dataset metadata, format conversion,
and common command help without requiring a separate LeRobot setup step.

If you already have a LeRobot checkout, you can still set `LEROBOT_ROOT=/path/to/lerobot` as an
advanced override.

Advanced install controls:

- Set `LEROBOT_MCP_LEROBOT_PYTHON=3.13` to use a different Python when preparing the managed
  checkout.
- Set `LEROBOT_MCP_LEROBOT_EXTRAS=dataset,core_scripts` to install more LeRobot extras by default.
- Pass `setup_environment=false` to `lerobot_install_or_update_lerobot` if you only want clone/update
  behavior.

### Codex

Add the local stdio MCP server with the Codex CLI:

```bash
codex mcp add lerobot-mcp -- lerobot-mcp
```

Or add it manually to `~/.codex/config.toml`:

```toml
[mcp_servers.lerobot_mcp]
command = "lerobot-mcp"
startup_timeout_sec = 20
tool_timeout_sec = 3600
```

Restart Codex after changing MCP config. In the Codex TUI, run `/mcp` to verify the server is loaded.
Then ask Codex: "Run `lerobot_install_or_update_lerobot`, then list LeRobot commands."

### Claude Code

Add the local stdio MCP server with Claude Code:

```bash
claude mcp add lerobot-mcp -- lerobot-mcp
```

If you are running from a checkout instead of a tool install:

```bash
claude mcp add lerobot-mcp -- /path/to/lerobot-mcp/.venv/bin/lerobot-mcp
```

After adding the server, restart Claude Code and run `/mcp` to verify the server is connected.
Then ask Claude: "Run `lerobot_install_or_update_lerobot`, then show `lerobot_capabilities`."

Resolution order is: `LEROBOT_ROOT`, current project ancestors, managed checkout
`~/.cache/lerobot-mcp/lerobot`, `~/hrl/lerobot`, then an installed `lerobot` package.

## Tool Model

The server does not expose arbitrary shell execution. It only runs:

- LeRobot entry points discovered from the configured checkout or installed distribution, such as
  `lerobot-train`, `lerobot-eval`, `lerobot-record`, `lerobot-replay`, `lerobot-annotate`,
  `lerobot-rollout`, and hardware setup utilities.
- Python scripts inside the configured LeRobot checkout's `examples/` directory.
- Pinned Forge commands via:

```bash
uv tool run --from "forge-robotics[hub,lerobot] @ git+https://github.com/arpitg1304/forge.git@461a0179115c7f2dc763ff4b1a1d2de02f5a1e69" forge ...
```

Options are passed as structured key/value pairs and serialized to draccus-compatible arguments:

```json
{
  "command": "train",
  "options": {
    "policy.type": "act",
    "dataset.repo_id": "lerobot/aloha_mobile_cabinet"
  }
}
```

That becomes:

```bash
uv run lerobot-train --dataset.repo_id=lerobot/aloha_mobile_cabinet --policy.type=act
```

## Main MCP Tools

- `lerobot_server_config`: show resolved LeRobot root, uv usage, managed Python/extras, and Forge pin.
- `lerobot_install_or_update_lerobot`: clone or update LeRobot `main` into the managed checkout and
  prepare its `uv` environment.
- `lerobot_list_commands`: list discovered LeRobot console scripts.
- `lerobot_capabilities`: audit current LeRobot commands, extras, examples, and registered components.
- `lerobot_public_symbols`: inspect public classes/functions below a LeRobot module prefix.
- `lerobot_command_help`: run `--help` for a discovered LeRobot command.
- `lerobot_list_examples`: list runnable examples in the checkout.
- `lerobot_build_command`: dry-run a command from structured options.
- `lerobot_build_example`: dry-run a LeRobot example script.
- `lerobot_run_command`: run a known LeRobot entry point.
- `lerobot_run_example`: run an example script under `examples/`.
- `lerobot_list_jobs`, `lerobot_job_status`, `lerobot_job_logs`, `lerobot_cancel_job`: manage
  background jobs.
- `lerobot_inspect_dataset_metadata`: summarize metadata for a local or Hub dataset.
- `lerobot_hf_whoami`, `lerobot_hf_repo_info`: Hugging Face Hub utilities.
- `lerobot_hf_search_datasets`: search datasets by robot, format, size, task, tags, and demo fit.
- `lerobot_build_dataset_latest_format_convert`, `lerobot_convert_dataset_to_latest_format`: convert
  LeRobot v2.1 datasets to the current v3.0 parquet layout.
- `lerobot_build_forge_convert`, `lerobot_forge_convert`: convert datasets with pinned Forge.
- `lerobot_build_forge_inspect`, `lerobot_forge_inspect`: inspect datasets with pinned Forge.

## LeRobot Dataset Format Migration

Latest LeRobot `main` currently uses the v3.0 parquet layout. The upstream converter supports v2.1
datasets and rewrites them to:

- `data/chunk-*/file_*.parquet`
- `videos/<camera>/chunk-*/file_*.mp4`
- `meta/tasks.parquet`
- `meta/episodes/chunk-*/file_*.parquet`
- aggregate `meta/stats.json`, with per-episode stats flattened into the episode parquet metadata

Preview a conversion:

```json
{
  "repo_id": "lerobot/berkeley_autolab_ur5",
  "root": "/tmp/berkeley_autolab_ur5",
  "force_conversion": true
}
```

Run it as a background job:

```json
{
  "repo_id": "lerobot/berkeley_autolab_ur5",
  "root": "/tmp/berkeley_autolab_ur5",
  "force_conversion": true,
  "background": true,
  "push_to_hub": false
}
```

`push_to_hub` defaults to `false`. For Hub datasets that already have a `v3.0` tag, omit
`force_conversion` to let the upstream script reuse the latest compatible version. Older branches such
as v1.x or v2.0 need to be brought to v2.1 before using this converter.

## Forge Conversion

Forge is pinned to commit `461a0179115c7f2dc763ff4b1a1d2de02f5a1e69` because the latest release does
not include the requested bug fixes.

Example MCP arguments:

```json
{
  "source": "hf://openvla/modified_libero_rlds",
  "output": "/tmp/libero_lerobot",
  "target_format": "lerobot-v3",
  "robot_type": "franka",
  "workers": 4,
  "dry_run": true
}
```

Equivalent command:

```bash
forge convert hf://openvla/modified_libero_rlds /tmp/libero_lerobot --format lerobot-v3 --robot-type franka --workers 4 --dry-run
```

## Dataset Search

Search is intended to help a user find datasets that fit their robot, computer, and target format.
It can combine Hugging Face Hub results with Forge registry metadata.

Example MCP arguments:

```json
{
  "query": "pusht",
  "robot": "aloha",
  "format": "lerobot",
  "max_size_gb": 10,
  "demo_suitable": true,
  "sort": "lastModified",
  "limit": 5
}
```

Results include source, repo id, detected format, robot hints, tags, scale when known, popularity
signals, and conversion hints.

For offline or deterministic tests, set `FORGE_REGISTRY_PATH` to a local `datasets.json` registry.

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run mypy
uv run pytest -vv
```

Run against latest LeRobot `main`:

```bash
cd /path/to/lerobot
git checkout main
git pull --ff-only origin main

cd /path/to/lerobot-mcp
LEROBOT_ROOT=/path/to/lerobot uv run lerobot-mcp
```

Build the package:

```bash
uv build
```

This repository is Apache-2.0 licensed.
