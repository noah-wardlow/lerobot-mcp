import json
from pathlib import Path
from typing import Any, ClassVar

from lerobot_mcp import hub
from lerobot_mcp.hub import _extract_robots, inspect_policy_repo
from lerobot_mcp.types import PolicyRepoInspectRequest


def test_extract_robots_matches_so101_aliases() -> None:
    values = [
        "JYeonKim/hil_lerobot_SO-101_wettissue_move_set_smolvla_ep100_re",
        "tuuy/lerobot_so_arm101_task0_new",
    ]

    assert _extract_robots(values) == ["so101"]


def test_inspect_policy_repo_summarizes_contract(monkeypatch, tmp_path: Path) -> None:
    config = {
        "type": "act",
        "fps": 30,
        "dataset": {
            "repo_id": "user/demo-dataset",
            "robot_type": "generic_arm",
        },
        "input_features": {
            "observation.images.front": {"type": "image", "shape": [3, 224, 224]},
            "observation.state": {"type": "float", "shape": [6]},
        },
        "output_features": {
            "action": {"type": "float", "shape": [6]},
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    class FakeSibling:
        def __init__(self, rfilename: str) -> None:
            self.rfilename = rfilename

    class FakeInfo:
        id = "user/demo-policy"
        sha = "abc123"
        private = False
        tags: ClassVar[list[str]] = ["lerobot", "act"]
        siblings: ClassVar[list[FakeSibling]] = [
            FakeSibling("config.json"),
            FakeSibling("model.safetensors"),
            FakeSibling("preprocessor_config.json"),
        ]

    class FakeHfApi:
        def repo_info(self, **_kwargs: Any) -> FakeInfo:
            return FakeInfo()

    def fake_download(**kwargs: Any) -> str:
        assert kwargs["repo_id"] == "user/demo-policy"
        assert kwargs["repo_type"] == "model"
        return str(config_path)

    monkeypatch.setattr(hub, "HfApi", FakeHfApi)
    monkeypatch.setattr(hub, "hf_hub_download", fake_download)

    result = inspect_policy_repo(
        PolicyRepoInspectRequest(repo_id="user/demo-policy", include_raw_configs=True)
    )

    assert result.repo_id == "user/demo-policy"
    assert result.policy_type == "act"
    assert result.dataset_repo_id == "user/demo-dataset"
    assert result.robot_type == "generic_arm"
    assert result.fps == 30.0
    assert result.image_keys == ["observation.images.front"]
    assert result.state_keys == ["observation.state"]
    assert result.action_keys == ["action"]
    assert result.missing_expected_files == []
    assert "config.json" in result.raw_configs


def test_inspect_policy_repo_reports_missing_contract_files(monkeypatch) -> None:
    class FakeInfo:
        id = "user/empty-policy"
        sha = "abc123"
        private = False
        tags: ClassVar[list[str]] = []
        siblings: ClassVar[list[object]] = []

    class FakeHfApi:
        def repo_info(self, **_kwargs: Any) -> FakeInfo:
            return FakeInfo()

    monkeypatch.setattr(hub, "HfApi", FakeHfApi)

    result = inspect_policy_repo(PolicyRepoInspectRequest(repo_id="user/empty-policy"))

    assert result.config_files == []
    assert result.weight_files == []
    assert result.missing_expected_files == ["config.json", "model weights"]
    assert "No explicit feature schema" in result.notes[-1]
