from pathlib import Path

import pytest

from lerobot_mcp.config import find_lerobot_root, install_or_update_lerobot, load_config, managed_lerobot_root
from lerobot_mcp.introspection import list_examples, resolve_example_path


def test_find_lerobot_root_uses_managed_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    managed = tmp_path / "cache" / "lerobot-mcp" / "lerobot"
    (managed / "src" / "lerobot").mkdir(parents=True)
    (managed / "pyproject.toml").write_text('[project]\nname="lerobot"\n')
    monkeypatch.delenv("LEROBOT_ROOT", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    assert managed_lerobot_root() == managed
    assert find_lerobot_root(start=tmp_path / "workspace") == managed.resolve()


def test_load_config_defaults_to_python_312_and_dataset_extra(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("LEROBOT_MCP_LEROBOT_PYTHON", raising=False)
    monkeypatch.delenv("LEROBOT_MCP_LEROBOT_EXTRAS", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    config = load_config()

    assert config.lerobot_python == "3.12"
    assert config.default_lerobot_extras == ("dataset",)


def test_load_config_accepts_custom_lerobot_environment_knobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LEROBOT_MCP_LEROBOT_PYTHON", "3.13")
    monkeypatch.setenv("LEROBOT_MCP_LEROBOT_EXTRAS", "dataset,core_scripts")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    config = load_config()

    assert config.lerobot_python == "3.13"
    assert config.default_lerobot_extras == ("dataset", "core_scripts")


def test_install_or_update_lerobot_syncs_managed_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[list[str], Path | None]] = []

    class Completed:
        def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(argv, cwd=None, **_kwargs):  # type: ignore[no-untyped-def]
        normalized = [str(arg) for arg in argv]
        calls.append((normalized, cwd))
        if normalized[:2] == ["git", "clone"]:
            target = Path(normalized[-1])
            (target / "src" / "lerobot").mkdir(parents=True)
            (target / ".git").mkdir()
            (target / "pyproject.toml").write_text('[project]\nname="lerobot"\n', encoding="utf-8")
        if normalized[:3] == ["git", "rev-parse", "HEAD"]:
            return Completed(stdout="abc123\n")
        return Completed()

    monkeypatch.setattr("lerobot_mcp.config.shutil.which", lambda name: "/bin/uv" if name == "uv" else None)
    monkeypatch.setattr("lerobot_mcp.config.subprocess.run", fake_run)

    result = install_or_update_lerobot(root=tmp_path / "lerobot", timeout_seconds=30)

    assert result["returncode"] == 0
    assert result["git_commit"] == "abc123"
    assert (
        ["/bin/uv", "sync", "--python", "3.12", "--extra", "dataset"],
        tmp_path / "lerobot",
    ) in calls
    assert result["environment_setup"] == {
        "requested": True,
        "ran": True,
        "uv_path": "/bin/uv",
        "python": "3.12",
        "extras": ["dataset"],
        "returncode": 0,
        "stdout": "",
        "stderr": "",
    }


def test_list_examples_filters_category(tmp_path: Path) -> None:
    examples = tmp_path / "examples"
    (examples / "training").mkdir(parents=True)
    (examples / "training" / "train_policy.py").write_text("print('ok')\n")
    (examples / "dataset").mkdir()
    (examples / "dataset" / "load.py").write_text("print('ok')\n")

    result = list_examples(examples, "training")

    assert [example.path for example in result] == ["training/train_policy.py"]
    assert result[0].category == "training"


def test_resolve_example_rejects_path_traversal(tmp_path: Path) -> None:
    examples = tmp_path / "examples"
    examples.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("print('no')\n")

    with pytest.raises(ValueError, match="inside"):
        resolve_example_path(examples, "../outside.py")
