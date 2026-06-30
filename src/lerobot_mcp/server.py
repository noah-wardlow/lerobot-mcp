from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from lerobot_mcp.config import (
    FORGE_COMMIT,
    FORGE_GIT_URL,
    FORGE_UV_SPEC,
    LEROBOT_GIT_URL,
    ServerConfig,
    discover_lerobot_roots,
    discover_project_scripts,
    get_git_commit,
    install_or_update_lerobot,
    load_config,
    managed_lerobot_root,
    validate_lerobot_root,
)
from lerobot_mcp.hub import inspect_policy_repo, search_datasets
from lerobot_mcp.introspection import discover_lerobot_capabilities, list_examples
from lerobot_mcp.metadata import inspect_dataset_metadata
from lerobot_mcp.runner import (
    ProcessManager,
    build_command_help_preview,
    build_entrypoint_preview,
    run_dataset_latest_format_convert,
    run_forge_convert,
    run_forge_inspect,
    run_lerobot_command,
    run_lerobot_example,
)
from lerobot_mcp.types import (
    CommandRequest,
    DatasetLatestFormatConvertRequest,
    DatasetMetadataRequest,
    DatasetSearchRequest,
    ExampleRequest,
    ForgeConvertRequest,
    ForgeInspectRequest,
    PolicyRepoInspectRequest,
)

CONFIG = load_config()
MANAGER = ProcessManager()

mcp = FastMCP(
    "LeRobot MCP",
    instructions="""Run and inspect LeRobot workflows.

Prefer dry-run tools before long-running robotics commands. Hardware commands can move real robots; ask the
user for the intended robot, ports, and workspace before running record, replay, teleoperate, calibrate,
setup-motors, or find-joint-limits. This server only executes LeRobot entry points discovered from the
configured checkout, scripts inside LeRobot examples, and dataset conversion helpers.
""",
)


def _auto_setup_enabled() -> bool:
    return os.getenv("LEROBOT_MCP_AUTO_SETUP", "1").lower() not in {"0", "false", "no"}


def _lerobot_config() -> ServerConfig:
    global CONFIG
    CONFIG = load_config()
    if CONFIG.lerobot_root is not None or not _auto_setup_enabled():
        return CONFIG
    result = install_or_update_lerobot(
        timeout_seconds=int(os.getenv("LEROBOT_MCP_AUTO_SETUP_TIMEOUT", "900")),
        setup_environment=True,
        python=CONFIG.lerobot_python,
        extras=CONFIG.default_lerobot_extras,
    )
    if result.get("returncode") != 0:
        setup = result.get("environment_setup")
        setup_stderr = setup.get("stderr", "") if isinstance(setup, dict) else ""
        stderr = str(result.get("stderr") or setup_stderr or "unknown error")
        raise RuntimeError(f"Could not prepare managed LeRobot checkout: {stderr}")
    CONFIG = load_config()
    return CONFIG


# --- Shared parameter annotations (drive JSON Schema `description` coverage) ---

OptionsParam = Annotated[
    dict[str, str | int | float | bool | None] | None,
    Field(
        description=(
            "LeRobot CLI options as structured key/value pairs, serialized to draccus "
            '`--key=value` arguments. Keys may be dotted paths, e.g. {"policy.type": "act", '
            '"dataset.repo_id": "lerobot/aloha_mobile_cabinet"}. Omit to pass no options.'
        )
    ),
]
ExtraArgsParam = Annotated[
    list[str] | None,
    Field(
        description=(
            "Raw CLI arguments appended verbatim after the generated options. Use for flags the "
            "structured options cannot express. Empty strings and NUL bytes are rejected."
        )
    ),
]
CwdParam = Annotated[
    str | None,
    Field(description="Working directory for the command. Defaults to the resolved LeRobot checkout root."),
]
UseUvParam = Annotated[
    bool,
    Field(
        description=(
            "Run via `uv run` inside the LeRobot environment when available; otherwise fall back to "
            "the resolved Python interpreter."
        )
    ),
]
EnvParam = Annotated[
    dict[str, str] | None,
    Field(
        description=(
            "Extra environment variables for the subprocess, merged over the inherited environment."
        )
    ),
]
BackgroundParam = Annotated[
    bool,
    Field(
        description=(
            "If true, start a managed background job and return immediately with a `job_id` to poll "
            "via lerobot_job_status/lerobot_job_logs. If false, block until the command finishes."
        )
    ),
]


@mcp.tool(
    annotations=ToolAnnotations(title="Show server configuration", readOnlyHint=True),
)
def lerobot_server_config() -> dict[str, Any]:
    """Return resolved server configuration and LeRobot checkout paths.

    Read-only. Call this first to see which LeRobot checkout is active, whether `uv` is available, and
    the managed Python/extras the server will use before running other tools.
    """
    global CONFIG
    CONFIG = load_config()
    return {
        "lerobot_root": str(CONFIG.lerobot_root) if CONFIG.lerobot_root else None,
        "examples_dir": str(CONFIG.examples_dir) if CONFIG.examples_dir else None,
        "uv_path": CONFIG.uv_path,
        "python_path": CONFIG.python_path,
        "prefer_uv": CONFIG.prefer_uv,
        "can_use_uv": CONFIG.can_use_uv,
        "lerobot_python": CONFIG.lerobot_python,
        "default_lerobot_extras": list(CONFIG.default_lerobot_extras),
        "auto_setup": _auto_setup_enabled(),
        "managed_lerobot_root": str(managed_lerobot_root()),
        "lerobot_git_url": LEROBOT_GIT_URL,
        "forge_git_url": FORGE_GIT_URL,
        "forge_commit": FORGE_COMMIT,
        "forge_uv_spec": FORGE_UV_SPEC,
    }


@mcp.tool(
    annotations=ToolAnnotations(title="Find LeRobot checkouts", readOnlyHint=True),
)
def lerobot_find_lerobot_roots(
    search_roots: Annotated[
        list[str] | None,
        Field(
            description=(
                "Directories to scan for LeRobot checkouts. Defaults to common locations: current "
                "project ancestors, the managed cache, and ~/hrl."
            )
        ),
    ] = None,
    max_depth: Annotated[
        int,
        Field(description="Maximum directory depth to descend while scanning each search root."),
    ] = 4,
    limit: Annotated[
        int,
        Field(description="Maximum number of checkouts to return."),
    ] = 20,
) -> list[dict[str, Any]]:
    """Find LeRobot checkouts the agent can select without requiring the user to set LEROBOT_ROOT.

    Read-only directory scan. Use this when no checkout is active, then pass a returned `path` to
    lerobot_use_lerobot_root to activate it for the session.
    """
    roots = discover_lerobot_roots(
        [Path(root).expanduser() for root in search_roots] if search_roots else None,
        max_depth=max_depth,
        limit=limit,
    )
    active = CONFIG.lerobot_root.resolve() if CONFIG.lerobot_root is not None else None
    managed = managed_lerobot_root()
    return [
        {
            "path": str(root),
            "active": active == root,
            "managed": root == managed,
            "git_commit": get_git_commit(root),
        }
        for root in roots
    ]


@mcp.tool(
    annotations=ToolAnnotations(
        title="Select LeRobot checkout",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
    ),
)
def lerobot_use_lerobot_root(
    root: Annotated[
        str,
        Field(
            description="Filesystem path to an existing LeRobot checkout to activate for this server process."
        ),
    ],
) -> dict[str, Any]:
    """Use an existing LeRobot checkout for this MCP server process.

    Changes which checkout subsequent tools resolve against; it does not modify any files. Pair with
    lerobot_find_lerobot_roots to discover candidate paths.
    """
    global CONFIG
    resolved = validate_lerobot_root(Path(root))
    os.environ["LEROBOT_ROOT"] = str(resolved)
    CONFIG = load_config()
    return {
        "lerobot_root": str(CONFIG.lerobot_root) if CONFIG.lerobot_root else None,
        "examples_dir": str(CONFIG.examples_dir) if CONFIG.examples_dir else None,
        "can_use_uv": CONFIG.can_use_uv,
        "git_commit": get_git_commit(CONFIG.lerobot_root),
    }


@mcp.tool(
    annotations=ToolAnnotations(
        title="Install or update LeRobot",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
def lerobot_install_or_update_lerobot(
    root: Annotated[
        str | None,
        Field(
            description="Target checkout path. Defaults to the managed cache at ~/.cache/lerobot-mcp/lerobot."
        ),
    ] = None,
    ref: Annotated[
        str,
        Field(description="Git ref (branch, tag, or commit) to check out. Defaults to `main`."),
    ] = "main",
    timeout_seconds: Annotated[
        int,
        Field(description="Maximum seconds allowed for the clone/update plus environment setup."),
    ] = 600,
    setup_environment: Annotated[
        bool,
        Field(description="Also create/sync the managed uv environment after fetching sources."),
    ] = True,
    python: Annotated[
        str | None,
        Field(
            description=(
                "Python version for the managed uv environment, e.g. '3.12'. Defaults to the server's "
                "configured version."
            )
        ),
    ] = None,
    extras: Annotated[
        list[str] | None,
        Field(
            description=(
                "LeRobot optional-dependency extras to install, e.g. ['dataset']. Defaults to the "
                "server's configured extras."
            )
        ),
    ] = None,
) -> dict[str, Any]:
    """Clone or fast-forward update LeRobot, then prepare a uv environment for common workflows.

    Writes to disk and reaches the network (git clone/pull). Safe to re-run; updates are fast-forward
    only. Most users should instead select an existing checkout with lerobot_use_lerobot_root.
    """
    global CONFIG
    result = install_or_update_lerobot(
        root=Path(root).expanduser() if root else None,
        ref=ref,
        timeout_seconds=timeout_seconds,
        setup_environment=setup_environment,
        python=python or CONFIG.lerobot_python,
        extras=extras if extras is not None else CONFIG.default_lerobot_extras,
    )
    CONFIG = load_config()
    result["active_lerobot_root"] = str(CONFIG.lerobot_root) if CONFIG.lerobot_root else None
    result["can_use_uv"] = CONFIG.can_use_uv
    return result


@mcp.tool(
    annotations=ToolAnnotations(title="List LeRobot commands", readOnlyHint=True),
)
def lerobot_list_commands() -> dict[str, Any]:
    """List supported command names and discovered LeRobot console scripts.

    Read-only. Returns the short command names (without the `lerobot-` prefix) accepted by
    lerobot_build_command/lerobot_run_command, plus the raw console scripts discovered in the checkout.
    """
    config = _lerobot_config()
    scripts = discover_project_scripts(config.lerobot_root)
    return {
        "commands": {
            name.removeprefix("lerobot-"): name for name in sorted(scripts) if name.startswith("lerobot-")
        },
        "discovered_console_scripts": scripts,
    }


@mcp.tool(
    annotations=ToolAnnotations(title="Audit LeRobot capabilities", readOnlyHint=True),
)
def lerobot_capabilities() -> dict[str, Any]:
    """List LeRobot scripts, extras, examples, and registered components from the current checkout.

    Read-only static inspection. Use this to discover which policies, robots, teleoperators, cameras,
    envs, and other registered components are available before building a command.
    """
    config = _lerobot_config()
    return discover_lerobot_capabilities(config.lerobot_root).model_dump(mode="json")


@mcp.tool(
    annotations=ToolAnnotations(title="List LeRobot examples", readOnlyHint=True),
)
def lerobot_list_examples(
    category: Annotated[
        str | None,
        Field(
            description=(
                "Optional example category (a subdirectory under examples/) to filter by, e.g. "
                "'getting_started'. Omit to list every example."
            )
        ),
    ] = None,
) -> list[dict[str, str]]:
    """List Python example scripts under the configured LeRobot checkout.

    Read-only. Returns example paths you can pass to lerobot_run_example.
    """
    config = _lerobot_config()
    return [example.model_dump(mode="json") for example in list_examples(config.examples_dir, category)]


@mcp.tool(
    annotations=ToolAnnotations(title="Preview LeRobot command", readOnlyHint=True),
)
def lerobot_build_command(
    command: Annotated[
        str,
        Field(
            description="LeRobot command name without the `lerobot-` prefix, e.g. 'train', 'eval', 'record'."
        ),
    ],
    options: OptionsParam = None,
    extra_args: ExtraArgsParam = None,
    cwd: CwdParam = None,
    use_uv: UseUvParam = True,
) -> dict[str, Any]:
    """Preview a known LeRobot entry point command without running it.

    Dry-run only: returns the exact argv, working directory, and environment that would be executed,
    but runs nothing. Use this to confirm a command before calling lerobot_run_command.
    """
    request = CommandRequest(
        command=command,
        options=options or {},
        extra_args=extra_args or [],
        cwd=Path(cwd).expanduser() if cwd else None,
        use_uv=use_uv,
    )
    return build_entrypoint_preview(_lerobot_config(), request).model_dump(mode="json")


@mcp.tool(
    annotations=ToolAnnotations(title="Show LeRobot command help", readOnlyHint=True, openWorldHint=True),
)
def lerobot_command_help(
    command: Annotated[
        str,
        Field(description="LeRobot console command name, with or without the `lerobot-` prefix."),
    ],
    timeout_seconds: Annotated[
        int,
        Field(description="Maximum seconds to wait for the `--help` subprocess."),
    ] = 60,
    use_uv: UseUvParam = True,
) -> dict[str, Any]:
    """Run `--help` for any discovered LeRobot console command.

    Read-only: runs the command with `--help` to capture its supported options. Use this to learn the
    option keys to pass to lerobot_build_command/lerobot_run_command.
    """
    preview = build_command_help_preview(_lerobot_config(), command, use_uv=use_uv)
    return MANAGER.run(preview, timeout_seconds=timeout_seconds, background=False).model_dump(mode="json")


@mcp.tool(
    annotations=ToolAnnotations(
        title="Run LeRobot command",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
def lerobot_run_command(
    command: Annotated[
        str,
        Field(
            description=(
                "LeRobot command name without the `lerobot-` prefix, e.g. 'train', 'eval', 'record', "
                "'replay'."
            )
        ),
    ],
    options: OptionsParam = None,
    extra_args: ExtraArgsParam = None,
    cwd: CwdParam = None,
    timeout_seconds: Annotated[
        int,
        Field(
            description=(
                "Maximum seconds before the run is terminated (foreground) or auto-stopped (background)."
            )
        ),
    ] = 900,
    background: BackgroundParam = True,
    use_uv: UseUvParam = True,
    env: EnvParam = None,
) -> dict[str, Any]:
    """Run a known LeRobot entry point as a foreground call or managed background job.

    Destructive: this can launch training, evaluation, recording, and hardware control that move real
    robots and write datasets/checkpoints. Confirm the robot, ports, and workspace with the user first,
    and preview with lerobot_build_command. Prefer `background=true` for long or hardware workflows.
    """
    request = CommandRequest(
        command=command,
        options=options or {},
        extra_args=extra_args or [],
        cwd=Path(cwd).expanduser() if cwd else None,
        timeout_seconds=timeout_seconds,
        background=background,
        use_uv=use_uv,
        env=env or {},
    )
    return run_lerobot_command(_lerobot_config(), MANAGER, request).model_dump(mode="json")


@mcp.tool(
    annotations=ToolAnnotations(
        title="Run LeRobot example",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
def lerobot_run_example(
    example_path: Annotated[
        str,
        Field(
            description=(
                "Path to an example script relative to the checkout's examples/ directory, e.g. "
                "'getting_started/1_load_lerobot_dataset.py'. Must resolve inside examples/."
            )
        ),
    ],
    options: OptionsParam = None,
    extra_args: ExtraArgsParam = None,
    timeout_seconds: Annotated[
        int,
        Field(
            description=(
                "Maximum seconds before the script is terminated (foreground) or auto-stopped (background)."
            )
        ),
    ] = 900,
    background: BackgroundParam = True,
    use_uv: UseUvParam = True,
    env: EnvParam = None,
) -> dict[str, Any]:
    """Run a Python script from LeRobot's examples directory.

    Destructive: examples may download data, train, or drive hardware. The path is constrained to the
    examples/ tree (path-traversal protected). List candidates with lerobot_list_examples first.
    """
    request = ExampleRequest(
        example_path=example_path,
        options=options or {},
        extra_args=extra_args or [],
        timeout_seconds=timeout_seconds,
        background=background,
        use_uv=use_uv,
        env=env or {},
    )
    return run_lerobot_example(_lerobot_config(), MANAGER, request).model_dump(mode="json")


@mcp.tool(
    annotations=ToolAnnotations(
        title="Convert dataset with Forge",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
def lerobot_forge_convert(
    source: Annotated[
        str,
        Field(description="Source dataset path or Hugging Face repo id to convert."),
    ],
    output: Annotated[
        str,
        Field(
            description=(
                "Destination directory for the converted dataset. Existing contents may be overwritten."
            )
        ),
    ],
    target_format: Annotated[
        str,
        Field(description="Target dataset format. Defaults to 'lerobot-v3'."),
    ] = "lerobot-v3",
    source_format: Annotated[
        str | None,
        Field(description="Source dataset format. Defaults to auto-detection."),
    ] = None,
    config_file: Annotated[
        str | None,
        Field(description="Optional path to a Forge conversion config file."),
    ] = None,
    fps: Annotated[
        float | None,
        Field(description="Frames per second to assign when the source lacks timing metadata. Must be > 0."),
    ] = None,
    robot_type: Annotated[
        str | None,
        Field(description="Robot/embodiment label to record in the converted dataset, e.g. 'so100'."),
    ] = None,
    camera_mapping: Annotated[
        dict[str, str] | None,
        Field(description="Mapping of source camera names to output observation image keys."),
    ] = None,
    workers: Annotated[
        int,
        Field(description="Number of parallel worker processes (1-256)."),
    ] = 1,
    fail_on_error: Annotated[
        bool,
        Field(
            description="Abort the whole conversion on the first per-episode error instead of skipping it."
        ),
    ] = False,
    visualize: Annotated[
        bool,
        Field(description="Render conversion previews where the converter supports it."),
    ] = False,
    dry_run: Annotated[
        bool,
        Field(description="Plan the conversion and report the actions without writing any output."),
    ] = False,
    background: BackgroundParam = True,
    timeout_seconds: Annotated[
        int,
        Field(
            description="Maximum seconds before the conversion is terminated (foreground) or auto-stopped."
        ),
    ] = 3_600,
    env: EnvParam = None,
) -> dict[str, Any]:
    """Run pinned Forge dataset conversion. Uses Forge main at the pinned bug-fix commit.

    Destructive: writes a converted dataset to `output` and may overwrite existing files. Use
    `dry_run=true` to preview, and lerobot_forge_inspect to check the source first. For LeRobot v2.1 to
    v3.0 upgrades specifically, prefer lerobot_convert_dataset_to_latest_format.
    """
    request = ForgeConvertRequest(
        source=source,
        output=Path(output).expanduser(),
        target_format=target_format,
        source_format=source_format,
        config_file=Path(config_file).expanduser() if config_file else None,
        fps=fps,
        robot_type=robot_type,
        camera_mapping=camera_mapping or {},
        workers=workers,
        fail_on_error=fail_on_error,
        visualize=visualize,
        dry_run=dry_run,
        background=background,
        timeout_seconds=timeout_seconds,
        env=env or {},
    )
    return run_forge_convert(CONFIG, MANAGER, request).model_dump(mode="json")


@mcp.tool(
    annotations=ToolAnnotations(title="Inspect dataset with Forge", readOnlyHint=True, openWorldHint=True),
)
def lerobot_forge_inspect(
    path: Annotated[
        str,
        Field(description="Dataset path or Hugging Face repo id to inspect."),
    ],
    format: Annotated[
        str | None,
        Field(description="Dataset format hint. Defaults to auto-detection."),
    ] = None,
    output: Annotated[
        str,
        Field(description="Report format: 'json' (default, structured) or 'text' (human-readable)."),
    ] = "json",
    quick: Annotated[
        bool,
        Field(description="Run a fast metadata-only inspection."),
    ] = True,
    deep: Annotated[
        bool,
        Field(description="Run a deeper inspection that reads sample records/frames."),
    ] = False,
    samples: Annotated[
        int,
        Field(description="Number of sample records to examine during deep inspection (1-1000)."),
    ] = 5,
    background: BackgroundParam = False,
    timeout_seconds: Annotated[
        int,
        Field(
            description="Maximum seconds before the inspection is terminated (foreground) or auto-stopped."
        ),
    ] = 600,
    env: EnvParam = None,
) -> dict[str, Any]:
    """Inspect a dataset with pinned Forge, useful before conversion.

    Read-only: reports a dataset's format, structure, and statistics without modifying it. Run this
    before lerobot_forge_convert to confirm the source format and detect issues.
    """
    request = ForgeInspectRequest(
        path=path,
        format=format,
        output="json" if output == "json" else "text",
        quick=quick,
        deep=deep,
        samples=samples,
        background=background,
        timeout_seconds=timeout_seconds,
        env=env or {},
    )
    return run_forge_inspect(CONFIG, MANAGER, request).model_dump(mode="json")


@mcp.tool(
    annotations=ToolAnnotations(
        title="Convert dataset to latest LeRobot format",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
def lerobot_convert_dataset_to_latest_format(
    repo_id: Annotated[
        str,
        Field(description="LeRobot dataset Hub repo id (e.g. 'lerobot/berkeley_autolab_ur5') to convert."),
    ],
    root: Annotated[
        str | None,
        Field(description="Local dataset root. Defaults to the Hub cache location for repo_id."),
    ] = None,
    branch: Annotated[
        str | None,
        Field(description="Source dataset branch/revision to convert from."),
    ] = None,
    data_file_size_in_mb: Annotated[
        int | None,
        Field(description="Target size per data parquet shard, in MB. Must be >= 1."),
    ] = None,
    video_file_size_in_mb: Annotated[
        int | None,
        Field(description="Target size per video shard, in MB. Must be >= 1."),
    ] = None,
    push_to_hub: Annotated[
        bool,
        Field(description="Upload the converted dataset back to the Hub. Defaults to false (local only)."),
    ] = False,
    force_conversion: Annotated[
        bool,
        Field(description="Convert from v2.1 even when a v3.0 tag already exists."),
    ] = False,
    background: BackgroundParam = True,
    timeout_seconds: Annotated[
        int,
        Field(
            description="Maximum seconds before the conversion is terminated (foreground) or auto-stopped."
        ),
    ] = 86_400,
    use_uv: UseUvParam = True,
    env: EnvParam = None,
) -> dict[str, Any]:
    """Convert a LeRobot v2.1 dataset to the current v3.0 parquet layout.

    Destructive: rewrites the dataset to the v3.0 layout and, when `push_to_hub=true`, uploads it.
    Defaults to local/no-push. Set `force_conversion=true` to convert from v2.1 even when a v3.0 tag
    already exists. Older Hub branches such as v1.x/v2.0 must first be migrated to v2.1. For non-LeRobot
    source formats, use lerobot_forge_convert instead.
    """
    request = DatasetLatestFormatConvertRequest(
        repo_id=repo_id,
        root=Path(root).expanduser() if root else None,
        branch=branch,
        data_file_size_in_mb=data_file_size_in_mb,
        video_file_size_in_mb=video_file_size_in_mb,
        push_to_hub=push_to_hub,
        force_conversion=force_conversion,
        background=background,
        timeout_seconds=timeout_seconds,
        use_uv=use_uv,
        env=env or {},
    )
    return run_dataset_latest_format_convert(_lerobot_config(), MANAGER, request).model_dump(mode="json")


JobIdParam = Annotated[
    str,
    Field(description="Job identifier returned when a tool was started with `background=true`."),
]


@mcp.tool(
    annotations=ToolAnnotations(title="Get background job status", readOnlyHint=True),
)
def lerobot_job_status(job_id: JobIdParam) -> dict[str, Any]:
    """Return state, return code, and output tails for a background job.

    Read-only. Poll this after starting a background run to check whether it is running, succeeded,
    failed, or was cancelled.
    """
    return MANAGER.status(job_id).model_dump(mode="json")


@mcp.tool(
    annotations=ToolAnnotations(title="List background jobs", readOnlyHint=True),
)
def lerobot_list_jobs() -> list[dict[str, Any]]:
    """List background jobs started during this MCP server process.

    Read-only. Jobs are tracked in-memory for the lifetime of the server process only.
    """
    return [job.model_dump(mode="json") for job in MANAGER.list()]


@mcp.tool(
    annotations=ToolAnnotations(title="Get background job logs", readOnlyHint=True),
)
def lerobot_job_logs(job_id: JobIdParam) -> dict[str, str]:
    """Return stdout and stderr tails for a background job.

    Read-only. Use this to inspect output from a job listed by lerobot_list_jobs.
    """
    return MANAGER.logs(job_id)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Cancel background job",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
    ),
)
def lerobot_cancel_job(job_id: JobIdParam) -> dict[str, Any]:
    """Terminate a background job.

    Destructive: sends a terminate signal to the running process. Idempotent — cancelling an
    already-finished job is a no-op that returns its final state.
    """
    return MANAGER.cancel(job_id).model_dump(mode="json")


@mcp.tool(
    annotations=ToolAnnotations(title="Inspect dataset metadata", readOnlyHint=True, openWorldHint=True),
)
def lerobot_inspect_dataset_metadata(
    repo_id: Annotated[
        str,
        Field(description="LeRobot dataset Hub repo id, or a local identifier when `root` is set."),
    ],
    root: Annotated[
        str | None,
        Field(
            description="Local dataset root path. When set, metadata is read locally instead of from the Hub."
        ),
    ] = None,
    revision: Annotated[
        str | None,
        Field(description="Dataset revision (branch, tag, or commit) to read."),
    ] = None,
    force_cache_sync: Annotated[
        bool,
        Field(description="Re-download metadata even if a cached copy already exists."),
    ] = False,
    timeout_seconds: Annotated[
        int,
        Field(description="Maximum seconds for the metadata read."),
    ] = 120,
    use_uv: UseUvParam = True,
) -> dict[str, Any]:
    """Summarize LeRobot dataset metadata from a local path or Hugging Face dataset repo.

    Read-only and lightweight: reads metadata (episodes, features, fps, format version) without
    importing heavy robotics dependencies. Use before converting or training to confirm a dataset fits.
    """
    request = DatasetMetadataRequest(
        repo_id=repo_id,
        root=Path(root).expanduser() if root else None,
        revision=revision,
        force_cache_sync=force_cache_sync,
        timeout_seconds=timeout_seconds,
        use_uv=use_uv,
    )
    result = inspect_dataset_metadata(_lerobot_config(), request)
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return result


@mcp.tool(
    annotations=ToolAnnotations(title="Search robotics datasets", readOnlyHint=True, openWorldHint=True),
)
def lerobot_hf_search_datasets(
    query: Annotated[
        str | None,
        Field(description="Free-text search over dataset names and descriptions."),
    ] = None,
    robot: Annotated[
        str | None,
        Field(description="Filter by robot/embodiment, e.g. 'aloha', 'so100'."),
    ] = None,
    format: Annotated[
        str | None,
        Field(description="Filter by dataset format, e.g. 'lerobot'."),
    ] = None,
    min_episodes: Annotated[
        int | None,
        Field(description="Minimum number of episodes. Must be >= 0."),
    ] = None,
    max_episodes: Annotated[
        int | None,
        Field(description="Maximum number of episodes. Must be >= 0."),
    ] = None,
    max_size_gb: Annotated[
        float | None,
        Field(description="Maximum on-disk size in gigabytes. Must be > 0."),
    ] = None,
    tags: Annotated[
        list[str] | None,
        Field(description="Hub tags that results must include."),
    ] = None,
    task: Annotated[
        str | None,
        Field(description="Filter by task type/skill, e.g. 'pick', 'pusht'."),
    ] = None,
    language_conditioned: Annotated[
        bool | None,
        Field(description="If set, require (true) or exclude (false) language-conditioned datasets."),
    ] = None,
    simulation: Annotated[
        bool | None,
        Field(description="If set, require (true) or exclude (false) simulation datasets."),
    ] = None,
    demo_suitable: Annotated[
        bool | None,
        Field(description="If true, prefer small datasets well suited to quick demos."),
    ] = None,
    prefer_lerobot: Annotated[
        bool,
        Field(description="Rank native LeRobot-format datasets higher."),
    ] = True,
    include_forge_registry: Annotated[
        bool,
        Field(description="Include results from the local Forge dataset registry (FORGE_REGISTRY_PATH)."),
    ] = True,
    include_hub: Annotated[
        bool,
        Field(description="Include results from the Hugging Face Hub."),
    ] = True,
    sort: Annotated[
        Literal["downloads", "likes", "lastModified", "createdAt", "trendingScore"],
        Field(description="Hub sort key for results."),
    ] = "downloads",
    limit: Annotated[
        int,
        Field(description="Maximum number of results to return (1-100)."),
    ] = 20,
) -> list[dict[str, Any]]:
    """Search robotics datasets by robot, format, scale, size, task, and compatibility hints.

    Read-only. Combines Hugging Face Hub results with the local Forge registry to help find a dataset
    that fits a given robot, machine, and target format. Returns conversion hints where relevant.
    """
    request = DatasetSearchRequest(
        query=query,
        robot=robot,
        format=format,
        min_episodes=min_episodes,
        max_episodes=max_episodes,
        max_size_gb=max_size_gb,
        tags=tags or [],
        task=task,
        language_conditioned=language_conditioned,
        simulation=simulation,
        demo_suitable=demo_suitable,
        prefer_lerobot=prefer_lerobot,
        include_forge_registry=include_forge_registry,
        include_hub=include_hub,
        sort=sort,
        limit=limit,
    )
    return [result.model_dump(mode="json") for result in search_datasets(request)]


@mcp.tool(
    annotations=ToolAnnotations(title="Inspect policy repo", readOnlyHint=True, openWorldHint=True),
)
def lerobot_inspect_policy_repo(
    repo_id: Annotated[
        str,
        Field(description="Hugging Face policy/model repo id to inspect, e.g. 'username/my-policy'."),
    ],
    revision: Annotated[
        str | None,
        Field(description="Repo revision (branch, tag, or commit) to read."),
    ] = None,
    include_raw_configs: Annotated[
        bool,
        Field(description="Include the raw parsed JSON config files in the result."),
    ] = False,
) -> dict[str, Any]:
    """Inspect a Hugging Face LeRobot policy/model repo for observation and action contract hints.

    Read-only and lightweight: reads repo metadata and JSON config files without importing LeRobot or
    running inference. Use before wiring a rollout to map camera/state keys and the action contract.
    """
    request = PolicyRepoInspectRequest(
        repo_id=repo_id,
        revision=revision,
        include_raw_configs=include_raw_configs,
    )
    return inspect_policy_repo(request).model_dump(mode="json")


def main() -> None:
    mcp.run()
