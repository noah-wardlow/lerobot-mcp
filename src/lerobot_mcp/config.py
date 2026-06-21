from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

LEROBOT_GIT_URL = "https://github.com/huggingface/lerobot.git"
FORGE_GIT_URL = "https://github.com/arpitg1304/forge.git"
FORGE_COMMIT = "461a0179115c7f2dc763ff4b1a1d2de02f5a1e69"
FORGE_UV_SPEC = f"forge-robotics[hub,lerobot] @ git+{FORGE_GIT_URL}@{FORGE_COMMIT}"
DEFAULT_LEROBOT_PYTHON = "3.12"
DEFAULT_LEROBOT_EXTRAS = ("dataset",)


@dataclass(frozen=True, slots=True)
class ServerConfig:
    lerobot_root: Path | None
    uv_path: str | None
    python_path: str
    prefer_uv: bool
    lerobot_python: str = DEFAULT_LEROBOT_PYTHON
    default_lerobot_extras: tuple[str, ...] = DEFAULT_LEROBOT_EXTRAS

    @property
    def examples_dir(self) -> Path | None:
        return self.lerobot_root / "examples" if self.lerobot_root is not None else None

    @property
    def can_use_uv(self) -> bool:
        return self.prefer_uv and self.uv_path is not None and self.lerobot_root is not None


def find_lerobot_root(start: Path | None = None) -> Path | None:
    env_root = os.getenv("LEROBOT_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        return root if _looks_like_lerobot_checkout(root) else None

    candidates: list[Path] = []
    if start is not None:
        candidates.extend([start.resolve(), *start.resolve().parents])
    candidates.append(Path.cwd().resolve())
    candidates.append(managed_lerobot_root())
    candidates.append(Path.home() / "hrl" / "lerobot")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _looks_like_lerobot_checkout(candidate):
            return candidate
    return None


def load_config() -> ServerConfig:
    return ServerConfig(
        lerobot_root=find_lerobot_root(),
        uv_path=shutil.which("uv"),
        python_path=sys.executable,
        prefer_uv=os.getenv("LEROBOT_MCP_NO_UV", "").lower() not in {"1", "true", "yes"},
        lerobot_python=os.getenv("LEROBOT_MCP_LEROBOT_PYTHON", DEFAULT_LEROBOT_PYTHON),
        default_lerobot_extras=_configured_lerobot_extras(),
    )


def managed_lerobot_root() -> Path:
    env_root = os.getenv("LEROBOT_MCP_MANAGED_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    cache_home = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    return (cache_home / "lerobot-mcp" / "lerobot").resolve()


def install_or_update_lerobot(
    *,
    root: Path | None = None,
    ref: str = "main",
    timeout_seconds: int = 600,
    setup_environment: bool = True,
    python: str | None = None,
    extras: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    target = (root or managed_lerobot_root()).expanduser().resolve()
    if target.exists() and not _looks_like_lerobot_checkout(target):
        raise ValueError(f"Target exists but is not a LeRobot checkout: {target}")

    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                "git",
                "clone",
                "--branch",
                ref,
                "--single-branch",
                LEROBOT_GIT_URL,
                str(target),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        action = "cloned"
    else:
        if not (target / ".git").exists():
            raise ValueError(f"LeRobot checkout is not a git checkout: {target}")
        completed = subprocess.run(
            ["git", "fetch", "origin", ref, "--prune"],
            cwd=target,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode == 0:
            completed = subprocess.run(
                ["git", "checkout", ref],
                cwd=target,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        if completed.returncode == 0:
            completed = subprocess.run(
                ["git", "pull", "--ff-only", "origin", ref],
                cwd=target,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        action = "updated"

    sync_result: subprocess.CompletedProcess[str] | None = None
    sync_extras = tuple(extras) if extras is not None else _configured_lerobot_extras()
    sync_python = (
        python if python is not None else os.getenv("LEROBOT_MCP_LEROBOT_PYTHON") or DEFAULT_LEROBOT_PYTHON
    )
    uv_path = shutil.which("uv")
    if completed.returncode == 0 and setup_environment and uv_path is not None:
        sync_result = sync_lerobot_environment(
            target,
            uv_path=uv_path,
            python=sync_python,
            extras=sync_extras,
            timeout_seconds=timeout_seconds,
        )

    returncode = completed.returncode
    if returncode == 0 and sync_result is not None and sync_result.returncode != 0:
        returncode = sync_result.returncode

    commit = get_git_commit(target) if completed.returncode == 0 else None
    return {
        "action": action,
        "root": str(target),
        "ref": ref,
        "returncode": returncode,
        "git_returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "git_commit": commit,
        "is_lerobot_checkout": _looks_like_lerobot_checkout(target),
        "environment_setup": {
            "requested": setup_environment,
            "ran": sync_result is not None,
            "uv_path": uv_path,
            "python": sync_python,
            "extras": list(sync_extras),
            "returncode": sync_result.returncode if sync_result is not None else None,
            "stdout": sync_result.stdout if sync_result is not None else "",
            "stderr": sync_result.stderr if sync_result is not None else "",
        },
    }


def sync_lerobot_environment(
    root: Path,
    *,
    uv_path: str,
    python: str = DEFAULT_LEROBOT_PYTHON,
    extras: tuple[str, ...] = DEFAULT_LEROBOT_EXTRAS,
    timeout_seconds: int = 600,
) -> subprocess.CompletedProcess[str]:
    argv = [uv_path, "sync", "--python", python]
    for extra in extras:
        argv.extend(["--extra", extra])
    return subprocess.run(
        argv,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def _looks_like_lerobot_checkout(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src" / "lerobot").is_dir()


def discover_project_scripts(root: Path | None) -> dict[str, str]:
    scripts: dict[str, str] = {}
    if root is not None:
        pyproject = root / "pyproject.toml"
        if pyproject.is_file():
            with pyproject.open("rb") as handle:
                data = tomllib.load(handle)
            project_scripts = data.get("project", {}).get("scripts", {})
            if isinstance(project_scripts, dict):
                scripts.update({str(name): str(target) for name, target in project_scripts.items()})

    try:
        dist = distribution("lerobot")
    except PackageNotFoundError:
        return scripts

    for entry_point in dist.entry_points:
        if entry_point.group == "console_scripts" and entry_point.name.startswith("lerobot-"):
            scripts.setdefault(entry_point.name, entry_point.value)
    return scripts


def discover_optional_dependencies(root: Path | None) -> dict[str, list[str]]:
    if root is None:
        return {}
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return {}
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    extras = data.get("project", {}).get("optional-dependencies", {})
    if not isinstance(extras, dict):
        return {}
    result: dict[str, list[str]] = {}
    for name, deps in extras.items():
        if isinstance(deps, list):
            result[str(name)] = [str(dep) for dep in deps]
    return result


def _configured_lerobot_extras() -> tuple[str, ...]:
    raw = os.getenv("LEROBOT_MCP_LEROBOT_EXTRAS")
    if raw is None:
        return DEFAULT_LEROBOT_EXTRAS
    extras = tuple(extra.strip() for extra in raw.split(",") if extra.strip())
    return extras


def resolve_lerobot_command(root: Path | None, command: str) -> str:
    scripts = discover_project_scripts(root)
    candidates = _command_candidates(command)
    for candidate in candidates:
        if candidate in scripts:
            return candidate
    raise ValueError(f"Unknown LeRobot command '{command}'. Available commands: {', '.join(sorted(scripts))}")


def command_shorthand(command_name: str) -> str:
    return command_name.removeprefix("lerobot-")


def get_git_commit(root: Path | None) -> str | None:
    if root is None or not (root / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _command_candidates(command: str) -> list[str]:
    normalized = command.strip()
    dashed = normalized.replace("_", "-")
    candidates = [normalized, dashed]
    if not dashed.startswith("lerobot-"):
        candidates.append(f"lerobot-{dashed}")
    return list(dict.fromkeys(candidates))
