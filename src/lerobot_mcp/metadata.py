from __future__ import annotations

import json
import subprocess
from typing import cast

from lerobot_mcp.config import ServerConfig
from lerobot_mcp.runner import _decode_timeout_output
from lerobot_mcp.types import DatasetMetadataRequest, ProcessResult


def inspect_dataset_metadata(
    config: ServerConfig,
    request: DatasetMetadataRequest,
) -> dict[str, object] | ProcessResult:
    code = """
import json
from pathlib import Path
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

repo_id = __REPO_ID__
root = __ROOT__
revision = __REVISION__
force_cache_sync = __FORCE_CACHE_SYNC__

meta = LeRobotDatasetMetadata(
    repo_id=repo_id,
    root=Path(root) if root is not None else None,
    revision=revision,
    force_cache_sync=force_cache_sync,
)
features = meta.info.get("features", {})
episodes = meta.episodes if meta.episodes is not None else []
tasks = meta.tasks if meta.tasks is not None else []
summary = {
    "repo_id": meta.repo_id,
    "root": str(meta.root),
    "revision": meta.revision,
    "codebase_version": meta.info.get("codebase_version"),
    "robot_type": meta.robot_type,
    "fps": meta.fps,
    "total_episodes": len(episodes),
    "total_tasks": len(tasks),
    "features": features,
    "data_path": meta.data_path,
    "video_path": meta.video_path,
}
print(json.dumps(summary, default=str))
""".replace("__REPO_ID__", repr(request.repo_id))
    code = code.replace("__ROOT__", repr(str(request.root) if request.root is not None else None))
    code = code.replace("__REVISION__", repr(request.revision))
    code = code.replace("__FORCE_CACHE_SYNC__", repr(request.force_cache_sync))

    if request.use_uv and config.can_use_uv:
        assert config.uv_path is not None
        argv = [config.uv_path, "run", "python", "-c", code]
        cwd = config.lerobot_root
    else:
        argv = [config.python_path, "-c", code]
        cwd = None

    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=request.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ProcessResult(
            argv=argv,
            cwd=cwd,
            returncode=124,
            stdout=_decode_timeout_output(exc.stdout),
            stderr=_decode_timeout_output(exc.stderr),
            timed_out=True,
        )

    if completed.returncode != 0:
        return ProcessResult(
            argv=argv,
            cwd=cwd,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    parsed = json.loads(completed.stdout)
    if not isinstance(parsed, dict):
        raise ValueError("LeRobot metadata helper returned non-object JSON")
    return cast(dict[str, object], parsed)
