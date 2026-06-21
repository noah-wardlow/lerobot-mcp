"""MCP server helpers for LeRobot."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lerobot-mcp")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
