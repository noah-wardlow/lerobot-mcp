from pathlib import Path

from lerobot_mcp import server
from lerobot_mcp.config import ServerConfig


def test_lerobot_config_lazily_prepares_managed_checkout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    missing = ServerConfig(
        lerobot_root=None,
        uv_path="/bin/uv",
        python_path="/bin/python",
        prefer_uv=True,
    )
    ready = ServerConfig(
        lerobot_root=tmp_path / "lerobot",
        uv_path="/bin/uv",
        python_path="/bin/python",
        prefer_uv=True,
    )
    calls: list[dict[str, object]] = []

    def fake_load_config() -> ServerConfig:
        return ready if calls else missing

    def fake_install_or_update_lerobot(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"returncode": 0}

    monkeypatch.setenv("LEROBOT_MCP_AUTO_SETUP", "1")
    monkeypatch.setattr(server, "CONFIG", missing)
    monkeypatch.setattr(server, "load_config", fake_load_config)
    monkeypatch.setattr(server, "install_or_update_lerobot", fake_install_or_update_lerobot)

    config = server._lerobot_config()

    assert config == ready
    assert calls == [
        {
            "timeout_seconds": 900,
            "setup_environment": True,
            "python": "3.12",
            "extras": ("dataset",),
        }
    ]


def test_lerobot_config_can_disable_managed_fallback(monkeypatch) -> None:
    missing = ServerConfig(
        lerobot_root=None,
        uv_path="/bin/uv",
        python_path="/bin/python",
        prefer_uv=True,
    )

    def fail_install_or_update_lerobot(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("managed fallback should be disabled")

    monkeypatch.setenv("LEROBOT_MCP_AUTO_SETUP", "0")
    monkeypatch.setattr(server, "CONFIG", missing)
    monkeypatch.setattr(server, "load_config", lambda: missing)
    monkeypatch.setattr(server, "install_or_update_lerobot", fail_install_or_update_lerobot)

    assert server._lerobot_config() == missing
