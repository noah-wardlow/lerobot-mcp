# lerobot-mcp

[![lerobot-mcp MCP server](https://glama.ai/mcp/servers/noah-wardlow/lerobot-mcp/badges/score.svg)](https://glama.ai/mcp/servers/noah-wardlow/lerobot-mcp)

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
- Inspect policy/model repo metadata for observation, image, state, and action contract hints.
- Optionally use Hub auth from your existing environment.
- Convert robotics datasets into LeRobot-compatible formats.
- Search datasets by robot, format, task, size, episode count, and compatibility hints.

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

With Docker:

```bash
docker build -t lerobot-mcp .
docker run -i --rm lerobot-mcp
```

The image runs the MCP server over stdio, so any MCP client can launch it with
`docker run -i --rm lerobot-mcp` as the command.

## MCP Quick Start

Most users should use the LeRobot checkout they already have. Start your MCP client from inside that
checkout, set `LEROBOT_ROOT=/path/to/lerobot` in the MCP server environment, or ask the agent to find
and select a checkout with `lerobot_find_lerobot_roots` and `lerobot_use_lerobot_root`.

If no checkout is found, LeRobot-backed tools lazily prepare a managed fallback at
`~/.cache/lerobot-mcp/lerobot` with Python 3.12 and LeRobot's `dataset` extra. That fallback covers
dataset metadata, format conversion, and common command help without requiring a separate setup step.

Advanced install controls:

- Set `LEROBOT_ROOT=/path/to/lerobot` to use a specific checkout.
- Set `LEROBOT_MCP_LEROBOT_PYTHON=3.13` to use a different Python when preparing the managed
  fallback.
- Set `LEROBOT_MCP_LEROBOT_EXTRAS=dataset,core_scripts` to install more LeRobot extras by default.
- Set `LEROBOT_MCP_AUTO_SETUP=0` to disable the managed fallback.

### Codex

Recommended:

```bash
codex mcp add lerobot-mcp -- lerobot-mcp
```

Manual fallback:

```toml
[mcp_servers.lerobot_mcp]
command = "lerobot-mcp"
startup_timeout_sec = 20
tool_timeout_sec = 3600
```

Restart Codex, run `/mcp`, then ask: "List LeRobot commands."

### Claude Code

```bash
claude mcp add lerobot-mcp -- lerobot-mcp
```

From a checkout:

```bash
claude mcp add lerobot-mcp -- /path/to/lerobot-mcp/.venv/bin/lerobot-mcp
```

Restart Claude Code, run `/mcp`, then ask: "Show `lerobot_capabilities`."

Resolution order is: `LEROBOT_ROOT`, current project ancestors, managed checkout
`~/.cache/lerobot-mcp/lerobot`, `~/hrl/lerobot`, then an installed `lerobot` package.

## Tool Model

The server does not expose arbitrary shell execution. It only runs:

- LeRobot entry points discovered from the configured checkout or installed distribution, such as
  `lerobot-train`, `lerobot-eval`, `lerobot-record`, `lerobot-replay`, `lerobot-annotate`,
  `lerobot-rollout`, and hardware setup utilities.
- Python scripts inside the configured LeRobot checkout's `examples/` directory.
- Dataset conversion helpers exposed by this MCP server.

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

- `lerobot_server_config`: show resolved LeRobot root, uv usage, and managed Python/extras.
- `lerobot_find_lerobot_roots`, `lerobot_use_lerobot_root`: find an existing LeRobot checkout and use
  it for the current MCP session.
- `lerobot_install_or_update_lerobot`: clone or update LeRobot `main` into the managed checkout and
  prepare its `uv` environment.
- `lerobot_list_commands`: list discovered LeRobot console scripts.
- `lerobot_capabilities`: audit current LeRobot commands, extras, examples, and registered components.
- `lerobot_command_help`: run `--help` for a discovered LeRobot command.
- `lerobot_list_examples`: list runnable examples in the checkout.
- `lerobot_build_command`: dry-run a command from structured options.
- `lerobot_run_command`: run a known LeRobot entry point.
- `lerobot_run_example`: run an example script under `examples/`.
- `lerobot_list_jobs`, `lerobot_job_status`, `lerobot_job_logs`, `lerobot_cancel_job`: manage
  background jobs.
- `lerobot_inspect_dataset_metadata`: summarize metadata for a local or Hub dataset.
- `lerobot_hf_search_datasets`: search datasets by robot, format, size, task, tags, and demo fit.
- `lerobot_inspect_policy_repo`: inspect a Hugging Face policy/model repo for config files, weights,
  policy type, dataset/robot hints, FPS, and declared observation/action features.
- `lerobot_convert_dataset_to_latest_format`: convert LeRobot v2.1 datasets to the current v3.0
  parquet layout.

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

## Dataset Search

Search is intended to help a user find datasets that fit their robot, computer, and target format.
It can combine Hub results with locally configured registry metadata.

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

## Policy Repo Inspection

Use policy inspection before wiring a real browser or simulator rollout. It does not import LeRobot or
run inference; it reads Hub repo metadata and lightweight JSON config files.

Example MCP arguments:

```json
{
  "repo_id": "username/my-policy",
  "include_raw_configs": false
}
```

The result includes config/weight file presence, policy type, dataset and robot hints, FPS, declared
input/output features, and classified `image_keys`, `state_keys`, and `action_keys`. Clients can use
that to map camera captures and state vectors before starting an inference server.

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
