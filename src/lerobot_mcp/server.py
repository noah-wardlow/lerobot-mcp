from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from lerobot_mcp.config import (
    FORGE_COMMIT,
    FORGE_GIT_URL,
    FORGE_UV_SPEC,
    discover_project_scripts,
    load_config,
)
from lerobot_mcp.hub import hf_repo_info, hf_whoami, search_datasets
from lerobot_mcp.introspection import discover_lerobot_capabilities, list_examples, module_public_symbols
from lerobot_mcp.metadata import inspect_dataset_metadata
from lerobot_mcp.runner import (
    ProcessManager,
    build_command_help_preview,
    build_dataset_latest_format_convert_preview,
    build_entrypoint_preview,
    build_example_preview,
    build_forge_convert_preview,
    build_forge_inspect_preview,
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
    RepoType,
)

CONFIG = load_config()
MANAGER = ProcessManager()

mcp = FastMCP(
    "LeRobot MCP",
    instructions="""Run and inspect Hugging Face LeRobot workflows.

Prefer dry-run tools before long-running robotics commands. Hardware commands can move real robots; ask the
user for the intended robot, ports, and workspace before running record, replay, teleoperate, calibrate,
setup-motors, or find-joint-limits. This server only executes LeRobot entry points discovered from the
configured checkout, scripts inside LeRobot examples, and pinned Forge commands for dataset conversion.
""",
)


@mcp.tool()
def lerobot_server_config() -> dict[str, Any]:
    """Return resolved server configuration and LeRobot checkout paths."""
    return {
        "lerobot_root": str(CONFIG.lerobot_root) if CONFIG.lerobot_root else None,
        "examples_dir": str(CONFIG.examples_dir) if CONFIG.examples_dir else None,
        "uv_path": CONFIG.uv_path,
        "python_path": CONFIG.python_path,
        "prefer_uv": CONFIG.prefer_uv,
        "can_use_uv": CONFIG.can_use_uv,
        "forge_git_url": FORGE_GIT_URL,
        "forge_commit": FORGE_COMMIT,
        "forge_uv_spec": FORGE_UV_SPEC,
    }


@mcp.tool()
def lerobot_list_commands() -> dict[str, Any]:
    """List supported command names and discovered LeRobot console scripts."""
    scripts = discover_project_scripts(CONFIG.lerobot_root)
    return {
        "commands": {
            name.removeprefix("lerobot-"): name
            for name in sorted(scripts)
            if name.startswith("lerobot-")
        },
        "discovered_console_scripts": scripts,
    }


@mcp.tool()
def lerobot_capabilities() -> dict[str, Any]:
    """List LeRobot scripts, extras, examples, and registered components from the current checkout."""
    return discover_lerobot_capabilities(CONFIG.lerobot_root).model_dump(mode="json")


@mcp.tool()
def lerobot_public_symbols(module_prefix: str = "lerobot", limit: int = 500) -> list[dict[str, str]]:
    """List public classes/functions below a LeRobot module prefix by static source inspection."""
    return module_public_symbols(CONFIG.lerobot_root, module_prefix)[:limit]


@mcp.tool()
def lerobot_list_examples(category: str | None = None) -> list[dict[str, str]]:
    """List Python example scripts under the configured LeRobot checkout."""
    return [example.model_dump(mode="json") for example in list_examples(CONFIG.examples_dir, category)]


@mcp.tool()
def lerobot_build_command(
    command: str,
    options: dict[str, str | int | float | bool | None] | None = None,
    extra_args: list[str] | None = None,
    cwd: str | None = None,
    use_uv: bool = True,
) -> dict[str, Any]:
    """Preview a known LeRobot entry point command without running it."""
    request = CommandRequest(
        command=command,
        options=options or {},
        extra_args=extra_args or [],
        cwd=Path(cwd).expanduser() if cwd else None,
        use_uv=use_uv,
    )
    return build_entrypoint_preview(CONFIG, request).model_dump(mode="json")


@mcp.tool()
def lerobot_command_help(
    command: str,
    timeout_seconds: int = 60,
    use_uv: bool = True,
) -> dict[str, Any]:
    """Run `--help` for any discovered LeRobot console command."""
    preview = build_command_help_preview(CONFIG, command, use_uv=use_uv)
    return MANAGER.run(preview, timeout_seconds=timeout_seconds, background=False).model_dump(mode="json")


@mcp.tool()
def lerobot_build_example(
    example_path: str,
    options: dict[str, str | int | float | bool | None] | None = None,
    extra_args: list[str] | None = None,
    use_uv: bool = True,
) -> dict[str, Any]:
    """Preview a LeRobot example script command without running it."""
    request = ExampleRequest(
        example_path=example_path,
        options=options or {},
        extra_args=extra_args or [],
        use_uv=use_uv,
    )
    return build_example_preview(CONFIG, request).model_dump(mode="json")


@mcp.tool()
def lerobot_run_command(
    command: str,
    options: dict[str, str | int | float | bool | None] | None = None,
    extra_args: list[str] | None = None,
    cwd: str | None = None,
    timeout_seconds: int = 900,
    background: bool = True,
    use_uv: bool = True,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run a known LeRobot entry point as a foreground call or managed background job."""
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
    return run_lerobot_command(CONFIG, MANAGER, request).model_dump(mode="json")


@mcp.tool()
def lerobot_run_example(
    example_path: str,
    options: dict[str, str | int | float | bool | None] | None = None,
    extra_args: list[str] | None = None,
    timeout_seconds: int = 900,
    background: bool = True,
    use_uv: bool = True,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run a Python script from LeRobot's examples directory."""
    request = ExampleRequest(
        example_path=example_path,
        options=options or {},
        extra_args=extra_args or [],
        timeout_seconds=timeout_seconds,
        background=background,
        use_uv=use_uv,
        env=env or {},
    )
    return run_lerobot_example(CONFIG, MANAGER, request).model_dump(mode="json")


@mcp.tool()
def lerobot_build_forge_convert(
    source: str,
    output: str,
    target_format: str = "lerobot-v3",
    source_format: str | None = None,
    config_file: str | None = None,
    fps: float | None = None,
    robot_type: str | None = None,
    camera_mapping: dict[str, str] | None = None,
    workers: int = 1,
    fail_on_error: bool = False,
    visualize: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Preview pinned Forge dataset conversion to or from LeRobot-compatible formats."""
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
    )
    return build_forge_convert_preview(CONFIG, request).model_dump(mode="json")


@mcp.tool()
def lerobot_forge_convert(
    source: str,
    output: str,
    target_format: str = "lerobot-v3",
    source_format: str | None = None,
    config_file: str | None = None,
    fps: float | None = None,
    robot_type: str | None = None,
    camera_mapping: dict[str, str] | None = None,
    workers: int = 1,
    fail_on_error: bool = False,
    visualize: bool = False,
    dry_run: bool = False,
    background: bool = True,
    timeout_seconds: int = 3_600,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run pinned Forge dataset conversion. Uses Forge main at the pinned bug-fix commit."""
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


@mcp.tool()
def lerobot_forge_inspect(
    path: str,
    format: str | None = None,
    output: str = "json",
    quick: bool = True,
    deep: bool = False,
    samples: int = 5,
    background: bool = False,
    timeout_seconds: int = 600,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Inspect a dataset with pinned Forge, useful before conversion."""
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


@mcp.tool()
def lerobot_build_forge_inspect(
    path: str,
    format: str | None = None,
    output: str = "json",
    quick: bool = True,
    deep: bool = False,
    samples: int = 5,
) -> dict[str, Any]:
    """Preview pinned Forge dataset inspection."""
    request = ForgeInspectRequest(
        path=path,
        format=format,
        output="json" if output == "json" else "text",
        quick=quick,
        deep=deep,
        samples=samples,
    )
    return build_forge_inspect_preview(CONFIG, request).model_dump(mode="json")


@mcp.tool()
def lerobot_build_dataset_latest_format_convert(
    repo_id: str,
    root: str | None = None,
    branch: str | None = None,
    data_file_size_in_mb: int | None = None,
    video_file_size_in_mb: int | None = None,
    push_to_hub: bool = False,
    force_conversion: bool = False,
    use_uv: bool = True,
) -> dict[str, Any]:
    """Preview LeRobot's official v2.1 -> current v3.0 parquet dataset conversion command."""
    request = DatasetLatestFormatConvertRequest(
        repo_id=repo_id,
        root=Path(root).expanduser() if root else None,
        branch=branch,
        data_file_size_in_mb=data_file_size_in_mb,
        video_file_size_in_mb=video_file_size_in_mb,
        push_to_hub=push_to_hub,
        force_conversion=force_conversion,
        use_uv=use_uv,
    )
    return build_dataset_latest_format_convert_preview(CONFIG, request).model_dump(mode="json")


@mcp.tool()
def lerobot_convert_dataset_to_latest_format(
    repo_id: str,
    root: str | None = None,
    branch: str | None = None,
    data_file_size_in_mb: int | None = None,
    video_file_size_in_mb: int | None = None,
    push_to_hub: bool = False,
    force_conversion: bool = False,
    background: bool = True,
    timeout_seconds: int = 86_400,
    use_uv: bool = True,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Convert a LeRobot v2.1 dataset to the current v3.0 parquet layout.

    Defaults to local/no-push behavior. Set `force_conversion=true` to convert from v2.1 even when a
    v3.0 tag already exists. Older Hub branches such as v1.x/v2.0 must first be migrated to v2.1.
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
    return run_dataset_latest_format_convert(CONFIG, MANAGER, request).model_dump(mode="json")


@mcp.tool()
def lerobot_job_status(job_id: str) -> dict[str, Any]:
    """Return state, return code, and output tails for a background job."""
    return MANAGER.status(job_id).model_dump(mode="json")


@mcp.tool()
def lerobot_list_jobs() -> list[dict[str, Any]]:
    """List background jobs started during this MCP server process."""
    return [job.model_dump(mode="json") for job in MANAGER.list()]


@mcp.tool()
def lerobot_job_logs(job_id: str) -> dict[str, str]:
    """Return stdout and stderr tails for a background job."""
    return MANAGER.logs(job_id)


@mcp.tool()
def lerobot_cancel_job(job_id: str) -> dict[str, Any]:
    """Terminate a background job."""
    return MANAGER.cancel(job_id).model_dump(mode="json")


@mcp.tool()
def lerobot_inspect_dataset_metadata(
    repo_id: str,
    root: str | None = None,
    revision: str | None = None,
    force_cache_sync: bool = False,
    timeout_seconds: int = 120,
    use_uv: bool = True,
) -> dict[str, Any]:
    """Summarize LeRobot dataset metadata from a local path or Hugging Face dataset repo."""
    request = DatasetMetadataRequest(
        repo_id=repo_id,
        root=Path(root).expanduser() if root else None,
        revision=revision,
        force_cache_sync=force_cache_sync,
        timeout_seconds=timeout_seconds,
        use_uv=use_uv,
    )
    result = inspect_dataset_metadata(CONFIG, request)
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return result


@mcp.tool()
def lerobot_hf_whoami() -> dict[str, Any]:
    """Return the Hugging Face account visible to huggingface_hub."""
    return hf_whoami()


@mcp.tool()
def lerobot_hf_repo_info(
    repo_id: str,
    repo_type: RepoType = RepoType.DATASET,
    revision: str | None = None,
) -> dict[str, Any]:
    """Return basic Hugging Face Hub repository information."""
    return hf_repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision).model_dump(mode="json")


@mcp.tool()
def lerobot_hf_search_datasets(
    query: str | None = None,
    robot: str | None = None,
    format: str | None = None,
    min_episodes: int | None = None,
    max_episodes: int | None = None,
    max_size_gb: float | None = None,
    tags: list[str] | None = None,
    task: str | None = None,
    language_conditioned: bool | None = None,
    simulation: bool | None = None,
    demo_suitable: bool | None = None,
    prefer_lerobot: bool = True,
    include_forge_registry: bool = True,
    include_hub: bool = True,
    sort: Literal["downloads", "likes", "lastModified", "createdAt", "trendingScore"] = "downloads",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search robotics datasets by robot, format, scale, size, task, and compatibility hints."""
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


def main() -> None:
    mcp.run()
