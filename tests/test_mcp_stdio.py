from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def _text_payload(result: Any) -> str:
    content = result.content
    assert content
    first = content[0]
    text = first.text
    assert isinstance(text, str)
    return text


@pytest.mark.anyio
async def test_mcp_stdio_lists_and_calls_safe_tools(tmp_path: Path) -> None:
    lerobot_root = tmp_path / "lerobot"
    (lerobot_root / "src" / "lerobot").mkdir(parents=True)
    (lerobot_root / "examples" / "training").mkdir(parents=True)
    (lerobot_root / "src" / "lerobot" / "__init__.py").write_text("", encoding="utf-8")
    (lerobot_root / "examples" / "training" / "train_policy.py").write_text("", encoding="utf-8")
    (lerobot_root / "pyproject.toml").write_text(
        """
[project]
name = "lerobot"
version = "0.0.0"

[project.scripts]
lerobot-train = "lerobot.scripts.train:main"
lerobot-record = "lerobot.scripts.record:main"
""".strip(),
        encoding="utf-8",
    )

    forge_registry = tmp_path / "forge-datasets.json"
    forge_registry.write_text(
        json.dumps(
            {
                "datasets": {
                    "test-pusht": {
                        "name": "Test PushT",
                        "format": "lerobot-v3",
                        "embodiment": ["aloha"],
                        "tags": ["simulation", "pusht"],
                        "task_types": ["manipulation"],
                        "demo_suitable": True,
                        "scale": {"episodes": 10, "hours": 0.2},
                        "sources": [{"type": "hf_hub", "uri": "lerobot/pusht"}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    env = {"LEROBOT_ROOT": str(lerobot_root), "FORGE_REGISTRY_PATH": str(forge_registry)}
    params = StdioServerParameters(
        command="uv",
        args=["run", "lerobot-mcp"],
        cwd=Path(__file__).parents[1],
        env=env,
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        tool_names = {tool.name for tool in tools.tools}

        assert "lerobot_capabilities" in tool_names
        assert "lerobot_find_lerobot_roots" in tool_names
        assert "lerobot_install_or_update_lerobot" in tool_names
        assert "lerobot_use_lerobot_root" in tool_names
        assert "lerobot_forge_inspect" in tool_names
        assert "lerobot_hf_search_datasets" in tool_names
        assert "lerobot_inspect_policy_repo" in tool_names
        assert "lerobot_list_jobs" in tool_names
        assert "lerobot_convert_dataset_to_latest_format" in tool_names
        assert "lerobot_build_dataset_latest_format_convert" not in tool_names
        assert "lerobot_build_example" not in tool_names
        assert "lerobot_build_forge_convert" not in tool_names
        assert "lerobot_build_forge_inspect" not in tool_names
        assert "lerobot_hf_repo_info" not in tool_names
        assert "lerobot_hf_whoami" not in tool_names
        assert "lerobot_public_symbols" not in tool_names

        config = await session.call_tool("lerobot_server_config")
        assert str(lerobot_root) in _text_payload(config)

        build = await session.call_tool(
            "lerobot_build_command",
            {"command": "train", "options": {"policy.type": "act"}},
        )
        assert "lerobot-train" in _text_payload(build)

        search = await session.call_tool(
            "lerobot_hf_search_datasets",
            {"query": "pusht", "format": "lerobot", "include_hub": False, "limit": 3},
        )
        parsed = search.structuredContent["result"]
        assert parsed
        assert any(item["source"] == "forge_registry" for item in parsed)
