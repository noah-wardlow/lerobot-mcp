from pathlib import Path

import pytest

from lerobot_mcp.config import ServerConfig
from lerobot_mcp.runner import (
    build_dataset_latest_format_convert_preview,
    build_entrypoint_preview,
    build_example_preview,
    build_forge_convert_preview,
    serialize_options,
)
from lerobot_mcp.types import (
    CommandRequest,
    DatasetLatestFormatConvertRequest,
    ExampleRequest,
    ForgeConvertRequest,
)


def test_serialize_options_uses_draccus_friendly_values() -> None:
    assert serialize_options({"b": True, "a": "act", "c": None, "d": 3}) == [
        "--a=act",
        "--b=true",
        "--c",
        "--d=3",
    ]


def test_command_options_reject_invalid_keys() -> None:
    with pytest.raises(ValueError, match="Invalid option key"):
        CommandRequest(command="train", options={"bad key": "value"})


def test_build_entrypoint_uses_uv_when_checkout_is_available(tmp_path: Path) -> None:
    (tmp_path / "src" / "lerobot").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="lerobot"\n[project.scripts]\nlerobot-train="lerobot.scripts.lerobot_train:main"\n'
    )
    config = ServerConfig(lerobot_root=tmp_path, uv_path="/bin/uv", python_path="/bin/python", prefer_uv=True)
    request = CommandRequest(command="train", options={"policy.type": "act"}, background=False)

    preview = build_entrypoint_preview(config, request)

    assert preview.argv == ["/bin/uv", "run", "lerobot-train", "--policy.type=act"]
    assert preview.cwd == tmp_path


def test_build_example_stays_under_examples_dir(tmp_path: Path) -> None:
    examples = tmp_path / "examples" / "training"
    examples.mkdir(parents=True)
    script = examples / "train_policy.py"
    script.write_text("print('ok')\n")
    config = ServerConfig(lerobot_root=tmp_path, uv_path=None, python_path="/bin/python", prefer_uv=True)
    request = ExampleRequest(example_path="training/train_policy.py", options={"dry_run": True})

    preview = build_example_preview(config, request)

    assert preview.argv == ["/bin/python", str(script), "--dry_run=true"]
    assert preview.cwd == tmp_path


def test_build_forge_convert_pins_main_commit(tmp_path: Path) -> None:
    config = ServerConfig(lerobot_root=tmp_path, uv_path="/bin/uv", python_path="/bin/python", prefer_uv=True)
    request = ForgeConvertRequest(
        source="hf://lerobot/pusht",
        output=tmp_path / "out",
        dry_run=True,
        camera_mapping={"agentview": "front"},
    )

    preview = build_forge_convert_preview(config, request)

    assert preview.argv[:5] == [
        "/bin/uv",
        "tool",
        "run",
        "--from",
        "forge-robotics[hub,lerobot] @ git+https://github.com/arpitg1304/forge.git@461a0179115c7f2dc763ff4b1a1d2de02f5a1e69",
    ]
    assert preview.argv[5:] == [
        "forge",
        "convert",
        "hf://lerobot/pusht",
        str(tmp_path / "out"),
        "--format",
        "lerobot-v3",
        "--workers",
        "1",
        "--camera",
        "agentview=front",
        "--dry-run",
    ]


def test_build_dataset_latest_format_convert_uses_official_lerobot_script(tmp_path: Path) -> None:
    script = tmp_path / "src" / "lerobot" / "scripts" / "convert_dataset_v21_to_v30.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('convert')\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname="lerobot"\n')
    config = ServerConfig(lerobot_root=tmp_path, uv_path="/bin/uv", python_path="/bin/python", prefer_uv=True)
    request = DatasetLatestFormatConvertRequest(
        repo_id="lerobot/berkeley_autolab_ur5",
        root=tmp_path / "datasets" / "berkeley_autolab_ur5",
        force_conversion=True,
        data_file_size_in_mb=50,
        video_file_size_in_mb=200,
    )

    preview = build_dataset_latest_format_convert_preview(config, request)

    assert preview.argv == [
        "/bin/uv",
        "run",
        "python",
        str(script),
        "--repo-id",
        "lerobot/berkeley_autolab_ur5",
        "--push-to-hub",
        "false",
        "--root",
        str(tmp_path / "datasets" / "berkeley_autolab_ur5"),
        "--data-file-size-in-mb",
        "50",
        "--video-file-size-in-mb",
        "200",
        "--force-conversion",
    ]
    assert preview.cwd == tmp_path
