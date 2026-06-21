from pathlib import Path

import pytest

from lerobot_mcp.config import find_lerobot_root, managed_lerobot_root
from lerobot_mcp.introspection import list_examples, resolve_example_path


def test_find_lerobot_root_uses_managed_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    managed = tmp_path / "cache" / "lerobot-mcp" / "lerobot"
    (managed / "src" / "lerobot").mkdir(parents=True)
    (managed / "pyproject.toml").write_text('[project]\nname="lerobot"\n')
    monkeypatch.delenv("LEROBOT_ROOT", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    assert managed_lerobot_root() == managed
    assert find_lerobot_root(start=tmp_path / "workspace") == managed.resolve()


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
