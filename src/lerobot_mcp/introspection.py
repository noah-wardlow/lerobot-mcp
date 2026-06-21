from __future__ import annotations

import ast
import re
from pathlib import Path

from lerobot_mcp.config import (
    command_shorthand,
    discover_optional_dependencies,
    discover_project_scripts,
    get_git_commit,
)
from lerobot_mcp.types import ExampleInfo, LeRobotCapabilities, LeRobotCommandInfo, RegistryItem


def list_examples(examples_dir: Path | None, category: str | None = None) -> list[ExampleInfo]:
    if examples_dir is None or not examples_dir.is_dir():
        return []

    normalized_category = category.strip("/") if category else None
    examples: list[ExampleInfo] = []
    for path in sorted(examples_dir.rglob("*.py")):
        relative = path.relative_to(examples_dir)
        if normalized_category is not None and not relative.as_posix().startswith(f"{normalized_category}/"):
            continue
        parts = relative.parts
        examples.append(
            ExampleInfo(
                path=relative.as_posix(),
                category=parts[0] if len(parts) > 1 else "",
                name=path.stem,
            )
        )
    return examples


def resolve_example_path(examples_dir: Path | None, example_path: str) -> Path:
    if examples_dir is None:
        raise ValueError("No LeRobot checkout configured. Set LEROBOT_ROOT to run examples.")
    root = examples_dir.resolve()
    candidate = (root / example_path).resolve()
    if candidate.suffix != ".py":
        raise ValueError("example_path must point to a Python file")
    if root != candidate and root not in candidate.parents:
        raise ValueError("example_path must stay inside the LeRobot examples directory")
    if not candidate.is_file():
        raise FileNotFoundError(f"Example not found: {example_path}")
    return candidate


def discover_lerobot_capabilities(root: Path | None) -> LeRobotCapabilities:
    commands = [
        _command_info(name, target)
        for name, target in sorted(discover_project_scripts(root).items())
        if name.startswith("lerobot-")
    ]
    examples = list_examples(root / "examples" if root is not None else None)
    registries = discover_registered_components(root)
    return LeRobotCapabilities(
        git_commit=get_git_commit(root),
        commands=commands,
        extras=discover_optional_dependencies(root),
        examples=examples,
        registries=registries,
    )


def discover_registered_components(root: Path | None) -> dict[str, list[RegistryItem]]:
    if root is None:
        return {}
    src = root / "src" / "lerobot"
    if not src.is_dir():
        return {}
    registries: dict[str, list[RegistryItem]] = {}
    for path in sorted(src.rglob("*.py")):
        rel = path.relative_to(root)
        text = path.read_text(errors="replace")
        for match in re.finditer(
            r"@(?P<registry>[A-Za-z_][A-Za-z0-9_]*)\.register_subclass\((?:name=)?[\"'](?P<name>[^\"']+)[\"']\)",
            text,
        ):
            class_name = _next_class_name(text, match.end())
            registry_name = match.group("registry")
            registries.setdefault(registry_name, []).append(
                RegistryItem(
                    name=match.group("name"),
                    registry=registry_name,
                    class_name=class_name,
                    module_path=_module_path_from_source(rel),
                    source_path=rel.as_posix(),
                )
            )
        for match in re.finditer(
            r"@ProcessorStepRegistry\.register\((?:name=)?[\"'](?P<name>[^\"']+)[\"']\)",
            text,
        ):
            registries.setdefault("ProcessorStepRegistry", []).append(
                RegistryItem(
                    name=match.group("name"),
                    registry="ProcessorStepRegistry",
                    class_name=_next_class_name(text, match.end()),
                    module_path=_module_path_from_source(rel),
                    source_path=rel.as_posix(),
                )
            )
    return {key: sorted(items, key=lambda item: item.name) for key, items in sorted(registries.items())}


def module_public_symbols(root: Path | None, module_prefix: str = "lerobot") -> list[dict[str, str]]:
    if root is None:
        return []
    package_root = root / "src" / module_prefix.replace(".", "/")
    if not package_root.is_dir():
        return []

    symbols: list[dict[str, str]] = []
    for path in sorted(package_root.rglob("*.py")):
        rel = path.relative_to(root)
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        module = _module_path_from_source(rel)
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue
                symbols.append(
                    {
                        "name": node.name,
                        "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                        "module": module,
                        "source_path": rel.as_posix(),
                    }
                )
    return symbols


def _command_info(name: str, target: str) -> LeRobotCommandInfo:
    module, _, function = target.partition(":")
    return LeRobotCommandInfo(
        name=name,
        target=target,
        module=module,
        function=function or "main",
        shorthand=command_shorthand(name),
    )


def _next_class_name(text: str, start: int) -> str | None:
    match = re.search(r"\nclass\s+([A-Za-z_][A-Za-z0-9_]*)", text[start:])
    return match.group(1) if match else None


def _module_path_from_source(relative_path: Path) -> str:
    parts = list(relative_path.with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    return ".".join(parts)
