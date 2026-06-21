from __future__ import annotations

import os
import subprocess
import tempfile
import time
import uuid
from collections.abc import Mapping
from pathlib import Path

from lerobot_mcp.config import FORGE_UV_SPEC, ServerConfig, resolve_lerobot_command
from lerobot_mcp.introspection import resolve_example_path
from lerobot_mcp.types import (
    CommandPreview,
    CommandRequest,
    DatasetLatestFormatConvertRequest,
    ExampleRequest,
    ForgeConvertRequest,
    ForgeInspectRequest,
    JobInfo,
    JobState,
    PrimitiveValue,
    ProcessResult,
)

TAIL_CHARS = 12_000


def serialize_options(options: Mapping[str, PrimitiveValue]) -> list[str]:
    args: list[str] = []
    for key in sorted(options):
        value = options[key]
        if value is None:
            args.append(f"--{key}")
        elif isinstance(value, bool):
            args.append(f"--{key}={str(value).lower()}")
        else:
            args.append(f"--{key}={value}")
    return args


def build_entrypoint_preview(config: ServerConfig, request: CommandRequest) -> CommandPreview:
    entrypoint = resolve_lerobot_command(config.lerobot_root, request.command)
    cwd = request.cwd or config.lerobot_root
    argv = [*_runtime_prefix(config, request.use_uv), entrypoint]
    argv.extend(serialize_options(request.options))
    argv.extend(request.extra_args)
    return CommandPreview(argv=argv, cwd=cwd, env=dict(request.env))


def build_command_help_preview(
    config: ServerConfig,
    command: str,
    *,
    use_uv: bool = True,
) -> CommandPreview:
    entrypoint = resolve_lerobot_command(config.lerobot_root, command)
    argv = [*_runtime_prefix(config, use_uv), entrypoint, "--help"]
    return CommandPreview(argv=argv, cwd=config.lerobot_root, env={})


def build_example_preview(config: ServerConfig, request: ExampleRequest) -> CommandPreview:
    example = resolve_example_path(config.examples_dir, request.example_path)
    cwd = config.lerobot_root
    runtime = _runtime_prefix(config, request.use_uv)
    argv = [*runtime, "python", str(example)] if runtime else [config.python_path, str(example)]
    argv.extend(serialize_options(request.options))
    argv.extend(request.extra_args)
    return CommandPreview(argv=argv, cwd=cwd, env=dict(request.env))


def build_forge_convert_preview(config: ServerConfig, request: ForgeConvertRequest) -> CommandPreview:
    argv = [
        *_forge_prefix(config),
        "convert",
        request.source,
        str(request.output),
        "--format",
        request.target_format,
        "--workers",
        str(request.workers),
    ]
    if request.config_file is not None:
        argv.extend(["--config", str(request.config_file)])
    if request.source_format is not None:
        argv.extend(["--source-format", request.source_format])
    if request.fps is not None:
        argv.extend(["--fps", str(request.fps)])
    if request.robot_type is not None:
        argv.extend(["--robot-type", request.robot_type])
    for source, target in sorted(request.camera_mapping.items()):
        argv.extend(["--camera", f"{source}={target}"])
    if request.fail_on_error:
        argv.append("--fail-on-error")
    if request.visualize:
        argv.append("--visualize")
    if request.dry_run:
        argv.append("--dry-run")
    return CommandPreview(argv=argv, cwd=config.lerobot_root, env=dict(request.env))


def build_forge_inspect_preview(config: ServerConfig, request: ForgeInspectRequest) -> CommandPreview:
    argv = [*_forge_prefix(config), "inspect", request.path, "--output", request.output]
    if request.format is not None:
        argv.extend(["--format", request.format])
    if request.quick:
        argv.append("--quick")
    if request.deep:
        argv.append("--deep")
    argv.extend(["--samples", str(request.samples)])
    return CommandPreview(argv=argv, cwd=config.lerobot_root, env=dict(request.env))


def build_dataset_latest_format_convert_preview(
    config: ServerConfig,
    request: DatasetLatestFormatConvertRequest,
) -> CommandPreview:
    if request.target_version != "v3.0":
        raise ValueError("LeRobot main currently exposes an official v2.1 -> v3.0 dataset converter only")

    script = (
        config.lerobot_root / "src" / "lerobot" / "scripts" / "convert_dataset_v21_to_v30.py"
        if config.lerobot_root is not None
        else None
    )
    if script is not None and script.is_file():
        runtime = _runtime_prefix(config, request.use_uv)
        argv = [*runtime, "python", str(script)] if runtime else [config.python_path, str(script)]
        cwd = config.lerobot_root
    else:
        runtime = _runtime_prefix(config, request.use_uv)
        argv = (
            [*runtime, "python", "-m", "lerobot.scripts.convert_dataset_v21_to_v30"]
            if runtime
            else [config.python_path, "-m", "lerobot.scripts.convert_dataset_v21_to_v30"]
        )
        cwd = config.lerobot_root

    argv.extend(["--repo-id", request.repo_id, "--push-to-hub", str(request.push_to_hub).lower()])
    if request.branch is not None:
        argv.extend(["--branch", request.branch])
    if request.root is not None:
        argv.extend(["--root", str(request.root)])
    if request.data_file_size_in_mb is not None:
        argv.extend(["--data-file-size-in-mb", str(request.data_file_size_in_mb)])
    if request.video_file_size_in_mb is not None:
        argv.extend(["--video-file-size-in-mb", str(request.video_file_size_in_mb)])
    if request.force_conversion:
        argv.append("--force-conversion")
    return CommandPreview(argv=argv, cwd=cwd, env=dict(request.env))


def _runtime_prefix(config: ServerConfig, use_uv: bool) -> list[str]:
    if use_uv and config.can_use_uv:
        assert config.uv_path is not None
        return [config.uv_path, "run"]
    return []


def _forge_prefix(config: ServerConfig) -> list[str]:
    if config.uv_path is None:
        raise RuntimeError("uv is required for pinned Forge integration but was not found on PATH")
    return [
        config.uv_path,
        "tool",
        "run",
        "--from",
        FORGE_UV_SPEC,
        "forge",
    ]


class ProcessManager:
    def __init__(self) -> None:
        self._jobs: dict[str, _ManagedJob] = {}

    def run(
        self,
        preview: CommandPreview,
        *,
        timeout_seconds: int,
        background: bool,
    ) -> ProcessResult | JobInfo:
        cwd = str(preview.cwd) if preview.cwd is not None else None
        env = _merged_env(preview.env, cwd=preview.cwd)
        if background:
            job_id = uuid.uuid4().hex
            stdout_fd, stdout_name = tempfile.mkstemp(prefix=f"lerobot-mcp-{job_id}-", suffix=".out")
            stderr_fd, stderr_name = tempfile.mkstemp(prefix=f"lerobot-mcp-{job_id}-", suffix=".err")
            with (
                os.fdopen(stdout_fd, "w", encoding="utf-8") as stdout_file,
                os.fdopen(stderr_fd, "w", encoding="utf-8") as stderr_file,
            ):
                proc = subprocess.Popen(
                    preview.argv,
                    cwd=cwd,
                    env=env,
                    stdout=stdout_file,
                    stderr=stderr_file,
                )
            job = _ManagedJob(
                job_id=job_id,
                proc=proc,
                preview=preview,
                started_at=time.time(),
                stdout_path=Path(stdout_name),
                stderr_path=Path(stderr_name),
            )
            self._jobs[job_id] = job
            return job.info()

        try:
            completed = subprocess.run(
                preview.argv,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            return ProcessResult(
                argv=preview.argv,
                cwd=preview.cwd,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            return ProcessResult(
                argv=preview.argv,
                cwd=preview.cwd,
                returncode=124,
                stdout=_decode_timeout_output(exc.stdout),
                stderr=_decode_timeout_output(exc.stderr),
                timed_out=True,
            )

    def status(self, job_id: str) -> JobInfo:
        return self._get(job_id).info()

    def list(self) -> list[JobInfo]:
        return [job.info() for job in self._jobs.values()]

    def logs(self, job_id: str) -> dict[str, str]:
        job = self._get(job_id)
        job.refresh()
        return {"stdout_tail": _read_tail(job.stdout_path), "stderr_tail": _read_tail(job.stderr_path)}

    def cancel(self, job_id: str) -> JobInfo:
        job = self._get(job_id)
        job.refresh()
        if job.proc.poll() is None:
            job.proc.terminate()
            try:
                job.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                job.proc.kill()
                job.proc.wait(timeout=5)
            job.cancelled = True
            job.finished_at = time.time()
        return job.info()

    def _get(self, job_id: str) -> _ManagedJob:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise KeyError(f"Unknown job_id: {job_id}") from exc


class _ManagedJob:
    def __init__(
        self,
        *,
        job_id: str,
        proc: subprocess.Popen[bytes],
        preview: CommandPreview,
        started_at: float,
        stdout_path: Path,
        stderr_path: Path,
    ) -> None:
        self.job_id = job_id
        self.proc = proc
        self.preview = preview
        self.started_at = started_at
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.finished_at: float | None = None
        self.cancelled = False

    def refresh(self) -> None:
        returncode = self.proc.poll()
        if returncode is None:
            return
        if self.finished_at is None:
            self.finished_at = time.time()

    def info(self) -> JobInfo:
        self.refresh()
        returncode = self.proc.poll()
        if self.cancelled:
            state = JobState.CANCELLED
        elif returncode is None:
            state = JobState.RUNNING
        elif returncode == 0:
            state = JobState.SUCCEEDED
        else:
            state = JobState.FAILED
        return JobInfo(
            job_id=self.job_id,
            state=state,
            argv=self.preview.argv,
            cwd=self.preview.cwd,
            returncode=returncode,
            started_at=self.started_at,
            finished_at=self.finished_at,
            stdout_tail=_read_tail(self.stdout_path),
            stderr_tail=_read_tail(self.stderr_path),
        )


def run_lerobot_command(
    config: ServerConfig,
    manager: ProcessManager,
    request: CommandRequest,
) -> ProcessResult | JobInfo:
    return manager.run(
        build_entrypoint_preview(config, request),
        timeout_seconds=request.timeout_seconds,
        background=request.background,
    )


def run_lerobot_example(
    config: ServerConfig,
    manager: ProcessManager,
    request: ExampleRequest,
) -> ProcessResult | JobInfo:
    return manager.run(
        build_example_preview(config, request),
        timeout_seconds=request.timeout_seconds,
        background=request.background,
    )


def run_forge_convert(
    config: ServerConfig,
    manager: ProcessManager,
    request: ForgeConvertRequest,
) -> ProcessResult | JobInfo:
    return manager.run(
        build_forge_convert_preview(config, request),
        timeout_seconds=request.timeout_seconds,
        background=request.background,
    )


def run_forge_inspect(
    config: ServerConfig,
    manager: ProcessManager,
    request: ForgeInspectRequest,
) -> ProcessResult | JobInfo:
    return manager.run(
        build_forge_inspect_preview(config, request),
        timeout_seconds=request.timeout_seconds,
        background=request.background,
    )


def run_dataset_latest_format_convert(
    config: ServerConfig,
    manager: ProcessManager,
    request: DatasetLatestFormatConvertRequest,
) -> ProcessResult | JobInfo:
    return manager.run(
        build_dataset_latest_format_convert_preview(config, request),
        timeout_seconds=request.timeout_seconds,
        background=request.background,
    )


def _merged_env(extra_env: Mapping[str, str], *, cwd: Path | None) -> dict[str, str]:
    env = dict(os.environ)
    active_venv = env.get("VIRTUAL_ENV")
    if cwd is not None and active_venv and not str(active_venv).startswith(str(cwd)):
        env.pop("VIRTUAL_ENV", None)
    env.update(extra_env)
    return env


def _tail(value: str) -> str:
    return value[-TAIL_CHARS:]


def _read_tail(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - TAIL_CHARS))
            return handle.read().decode(errors="replace")
    except FileNotFoundError:
        return ""


def _decode_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value
