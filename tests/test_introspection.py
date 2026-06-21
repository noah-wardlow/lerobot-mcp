from pathlib import Path

import pytest

from lerobot_mcp.introspection import list_examples, resolve_example_path


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
