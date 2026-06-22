from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Literal, cast

from huggingface_hub import HfApi, hf_hub_download

from lerobot_mcp.config import FORGE_COMMIT
from lerobot_mcp.types import (
    DatasetSearchRequest,
    DatasetSearchResult,
    HubRepoInfo,
    JsonValue,
    PolicyRepoInspection,
    PolicyRepoInspectRequest,
    RepoType,
)

type HfDatasetSort = Literal["created_at", "downloads", "last_modified", "likes", "trending_score"]

POLICY_CONFIG_CANDIDATES = (
    "config.json",
    "policy_config.json",
    "train_config.json",
    "preprocessor_config.json",
)
POLICY_PROCESSOR_FILES = {
    "preprocessor_config.json",
    "processor_config.json",
    "normalization_stats.json",
    "stats.json",
}
POLICY_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")


def hf_whoami() -> dict[str, object]:
    return HfApi().whoami()


def hf_repo_info(
    repo_id: str,
    repo_type: RepoType = RepoType.DATASET,
    revision: str | None = None,
) -> HubRepoInfo:
    api = HfApi()
    info = api.repo_info(repo_id=repo_id, repo_type=repo_type.value, revision=revision)
    siblings_raw = cast(Any, getattr(info, "siblings", None)) or []
    siblings = [str(sibling.rfilename) for sibling in siblings_raw]
    return HubRepoInfo(
        repo_id=str(info.id),
        repo_type=repo_type,
        private=getattr(info, "private", None),
        sha=getattr(info, "sha", None),
        tags=list(getattr(info, "tags", []) or []),
        siblings=siblings,
    )


def inspect_policy_repo(request: PolicyRepoInspectRequest) -> PolicyRepoInspection:
    api = HfApi()
    info = api.repo_info(
        repo_id=request.repo_id,
        repo_type=RepoType.MODEL.value,
        revision=request.revision,
    )
    siblings_raw = cast(Any, getattr(info, "siblings", None)) or []
    files = sorted(str(sibling.rfilename) for sibling in siblings_raw)
    config_files = [name for name in files if name in POLICY_CONFIG_CANDIDATES]
    weight_files = [name for name in files if name.endswith(POLICY_WEIGHT_SUFFIXES)]
    processor_files = [name for name in files if name in POLICY_PROCESSOR_FILES]
    configs = _download_policy_configs(request.repo_id, config_files, request.revision)
    merged = _merge_config_objects(configs)
    input_features = _feature_dict(
        _first_mapping(merged, ("input_features", "observation_features", "features"))
    )
    output_features = _feature_dict(_first_mapping(merged, ("output_features", "action_features")))
    image_keys = _classify_feature_keys(input_features, "image")
    state_keys = _classify_feature_keys(input_features, "state")
    action_keys = _classify_feature_keys(output_features, "action")
    missing_expected_files: list[str] = []
    if not config_files:
        missing_expected_files.append("config.json")
    if not weight_files:
        missing_expected_files.append("model weights")

    notes: list[str] = []
    if image_keys:
        notes.append(
            "Image observation keys are declared; browser clients should map camera captures by name."
        )
    if state_keys:
        notes.append(
            "State observation keys are declared; clients should verify qpos/qvel ordering against training."
        )
    if action_keys:
        notes.append(
            "Action keys are declared; clients should verify actuator ordering and scaling before rollout."
        )
    if not input_features and not output_features:
        notes.append("No explicit feature schema found in lightweight config files.")

    return PolicyRepoInspection(
        repo_id=str(info.id),
        revision=request.revision,
        sha=getattr(info, "sha", None),
        private=getattr(info, "private", None),
        tags=list(getattr(info, "tags", []) or []),
        config_files=config_files,
        weight_files=weight_files,
        processor_files=processor_files,
        policy_type=_first_string(merged, ("policy_type", "type"), path_contains=("policy", "config")),
        dataset_repo_id=_first_string(merged, ("dataset_repo_id", "repo_id"), path_contains=("dataset",)),
        robot_type=_first_string(merged, ("robot_type",), path_contains=("robot", "dataset")),
        fps=_first_float(merged, ("fps",)),
        input_features=input_features,
        output_features=output_features,
        image_keys=image_keys,
        state_keys=state_keys,
        action_keys=action_keys,
        missing_expected_files=missing_expected_files,
        notes=notes,
        raw_configs=configs if request.include_raw_configs else {},
    )


def search_datasets(request: DatasetSearchRequest) -> list[DatasetSearchResult]:
    results: list[DatasetSearchResult] = []
    if request.include_forge_registry:
        results.extend(_search_forge_registry(request))
    if request.include_hub:
        results.extend(_search_hub_datasets(request))

    deduped: dict[str, DatasetSearchResult] = {}
    for result in _sort_results(results, request):
        key = result.repo_id or result.id
        if key not in deduped:
            deduped[key] = result
    return list(deduped.values())[: request.limit]


def _search_hub_datasets(request: DatasetSearchRequest) -> list[DatasetSearchResult]:
    api = HfApi()
    query = request.query or " ".join(part for part in [request.robot, request.format, request.task] if part)
    if not query:
        query = "lerobot robotics"
    tags = list(request.tags)
    if request.language_conditioned:
        tags.append("language_conditioned")
    if request.simulation is True:
        tags.append("simulation")
    elif request.simulation is False:
        tags.append("real_world")

    limit = min(max(request.limit * 4, 20), 100)
    try:
        infos = api.list_datasets(
            search=query,
            filter=tags or None,
            sort=_hf_dataset_sort(request.sort),
            limit=limit,
            full=True,
        )
    except TypeError:
        infos = api.list_datasets(search=query, sort=_hf_dataset_sort(request.sort), limit=limit, full=True)

    results: list[DatasetSearchResult] = []
    for info in infos:
        repo_id = str(info.id)
        raw_tags = [str(tag) for tag in (getattr(info, "tags", []) or [])]
        siblings = cast(Any, getattr(info, "siblings", None)) or []
        sibling_names = [str(getattr(sibling, "rfilename", "")) for sibling in siblings]
        size_bytes = sum(int(getattr(sibling, "size", 0) or 0) for sibling in siblings)
        detected_format = _detect_dataset_format(raw_tags, sibling_names, repo_id)
        robots = _extract_robots([*raw_tags, repo_id])
        if not _matches_request(
            request,
            fmt=detected_format,
            robots=robots,
            tags=raw_tags,
            episodes=None,
            size_bytes=size_bytes or None,
        ):
            continue
        score = _score_result(
            request,
            text=" ".join([repo_id, detected_format or "", " ".join(raw_tags)]),
            fmt=detected_format,
            robots=robots,
            tags=raw_tags,
            downloads=getattr(info, "downloads", None),
            likes=getattr(info, "likes", None),
        )
        results.append(
            DatasetSearchResult(
                id=repo_id,
                name=repo_id.split("/")[-1],
                source="hf",
                repo_id=repo_id,
                format=detected_format,
                robot=robots,
                tags=raw_tags,
                size_bytes=size_bytes or None,
                downloads=getattr(info, "downloads", None),
                likes=getattr(info, "likes", None),
                private=getattr(info, "private", None),
                created_at=_stringify_datetime(getattr(info, "created_at", None)),
                last_modified=_stringify_datetime(getattr(info, "last_modified", None)),
                score=score,
                conversion_hint=_conversion_hint(detected_format, repo_id),
            )
        )
    return results


def _search_forge_registry(request: DatasetSearchRequest) -> list[DatasetSearchResult]:
    datasets = _load_forge_registry()
    results: list[DatasetSearchResult] = []
    for dataset_id, entry in datasets.items():
        fmt = str(entry.get("format", ""))
        robots = [str(robot) for robot in entry.get("embodiment", [])]
        tags = [str(tag) for tag in entry.get("tags", [])]
        task_types = [str(task) for task in entry.get("task_types", [])]
        scale = entry.get("scale") or {}
        episodes = scale.get("episodes") if isinstance(scale, dict) else None
        hours = scale.get("hours") if isinstance(scale, dict) else None
        if not _matches_request(
            request,
            fmt=fmt,
            robots=robots,
            tags=[*tags, *task_types],
            episodes=int(episodes) if isinstance(episodes, int) else None,
            size_bytes=None,
        ):
            continue
        if (
            request.demo_suitable is not None
            and bool(entry.get("demo_suitable")) is not request.demo_suitable
        ):
            continue
        sources = entry.get("sources") or []
        repo_id = _first_hf_source(sources)
        text = " ".join(
            [
                dataset_id,
                str(entry.get("name", "")),
                str(entry.get("description", "")),
                fmt,
                " ".join(robots),
                " ".join(tags),
                " ".join(task_types),
            ]
        )
        score = _score_result(
            request,
            text=text,
            fmt=fmt,
            robots=robots,
            tags=[*tags, *task_types],
            downloads=None,
            likes=None,
        )
        results.append(
            DatasetSearchResult(
                id=dataset_id,
                name=str(entry.get("name", dataset_id)),
                source="forge_registry",
                repo_id=repo_id,
                format=fmt,
                robot=robots,
                tags=tags,
                task_types=task_types,
                episodes=episodes if isinstance(episodes, int) else None,
                hours=float(hours) if isinstance(hours, int | float) else None,
                score=score,
                conversion_hint=_conversion_hint(fmt, dataset_id),
                notes=str(entry.get("description", "")) or None,
            )
        )
    return results


def _download_policy_configs(
    repo_id: str,
    config_files: list[str],
    revision: str | None,
) -> dict[str, JsonValue]:
    configs: dict[str, JsonValue] = {}
    for filename in config_files:
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=RepoType.MODEL.value,
                revision=revision,
            )
        except Exception:
            continue
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        if _is_json_value(value):
            configs[filename] = value
    return configs


def _merge_config_objects(configs: dict[str, JsonValue]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value in configs.values():
        if isinstance(value, dict):
            _deep_merge(merged, value)
    return merged


def _deep_merge(target: dict[str, Any], source: dict[str, JsonValue]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(cast(dict[str, Any], target[key]), value)
        else:
            target[key] = value


def _first_mapping(config: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    for path, value in _walk_config(config):
        if path and path[-1] in keys and isinstance(value, dict):
            return value
    return {}


def _feature_dict(value: dict[str, Any]) -> dict[str, JsonValue]:
    return {
        str(key): cast(JsonValue, feature)
        for key, feature in value.items()
        if _is_json_value(feature)
    }


def _classify_feature_keys(features: dict[str, JsonValue], kind: str) -> list[str]:
    matches: list[str] = []
    for key, value in features.items():
        haystack = f"{key} {json.dumps(value, sort_keys=True)}".lower()
        if kind in haystack:
            matches.append(key)
    return matches


def _first_string(
    config: dict[str, Any],
    keys: tuple[str, ...],
    *,
    path_contains: tuple[str, ...] = (),
) -> str | None:
    fallback: str | None = None
    for path, value in _walk_config(config):
        if not path or path[-1] not in keys or not isinstance(value, str):
            continue
        if not path_contains or any(part in ".".join(path[:-1]).lower() for part in path_contains):
            return value
        fallback = fallback or value
    return fallback


def _first_float(config: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for path, value in _walk_config(config):
        if not path or path[-1] not in keys:
            continue
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return None


def _walk_config(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    items = [(path, value)]
    if isinstance(value, dict):
        for key, child in value.items():
            items.extend(_walk_config(child, (*path, str(key))))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            items.extend(_walk_config(child, (*path, str(index))))
    return items


def _is_json_value(value: Any) -> bool:
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


def _load_forge_registry() -> dict[str, dict[str, Any]]:
    candidates = [os.getenv("FORGE_REGISTRY_PATH")]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            with open(candidate) as handle:
                data = json.load(handle)
            datasets = data.get("datasets", {})
            if isinstance(datasets, dict):
                return cast(dict[str, dict[str, Any]], datasets)

    url = (
        "https://raw.githubusercontent.com/arpitg1304/forge/"
        f"{FORGE_COMMIT}/forge/registry/datasets.json"
    )
    with urllib.request.urlopen(url, timeout=15) as response:
        data = json.loads(response.read().decode())
    datasets = data.get("datasets", {})
    return cast(dict[str, dict[str, Any]], datasets if isinstance(datasets, dict) else {})


def _matches_request(
    request: DatasetSearchRequest,
    *,
    fmt: str | None,
    robots: list[str],
    tags: list[str],
    episodes: int | None,
    size_bytes: int | None,
) -> bool:
    if request.format and (fmt is None or request.format.lower() not in fmt.lower()):
        return False
    if request.robot:
        robot = _normalize_robot_text(request.robot)
        if not any(robot in _normalize_robot_text(candidate) for candidate in robots + tags):
            return False
    if request.min_episodes is not None and (episodes is None or episodes < request.min_episodes):
        return False
    if request.max_episodes is not None and episodes is not None and episodes > request.max_episodes:
        return False
    if (
        request.max_size_gb is not None
        and size_bytes is not None
        and size_bytes > request.max_size_gb * 1_000_000_000
    ):
        return False
    for tag in request.tags:
        if not any(tag.lower() in candidate.lower() for candidate in tags):
            return False
    if request.task and not any(request.task.lower() in candidate.lower() for candidate in tags):
        return False
    if request.language_conditioned is not None:
        has_language = any("language" in tag.lower() for tag in tags)
        if has_language is not request.language_conditioned:
            return False
    if request.simulation is not None:
        has_sim = any("simulation" in tag.lower() for tag in tags)
        has_real = any("real_world" in tag.lower() or "real world" in tag.lower() for tag in tags)
        if request.simulation and not has_sim:
            return False
        if not request.simulation and has_sim and not has_real:
            return False
    return True


def _score_result(
    request: DatasetSearchRequest,
    *,
    text: str,
    fmt: str | None,
    robots: list[str],
    tags: list[str],
    downloads: int | None,
    likes: int | None,
) -> float:
    score = 0.0
    haystack = text.lower()
    for term in [request.query, request.robot, request.format, request.task, *request.tags]:
        if term and term.lower() in haystack:
            score += 10.0
    if request.prefer_lerobot and fmt and "lerobot" in fmt.lower():
        score += 8.0
    if request.robot and any(request.robot.lower() == robot.lower() for robot in robots):
        score += 5.0
    if request.language_conditioned and any("language" in tag.lower() for tag in tags):
        score += 4.0
    if downloads:
        score += min(downloads / 10_000, 10.0)
    if likes:
        score += min(likes / 20, 5.0)
    return score


def _sort_results(
    results: list[DatasetSearchResult],
    request: DatasetSearchRequest,
) -> list[DatasetSearchResult]:
    if request.sort == "lastModified":
        return sorted(results, key=lambda item: item.last_modified or "", reverse=True)
    if request.sort == "createdAt":
        return sorted(results, key=lambda item: item.created_at or "", reverse=True)
    if request.sort == "likes":
        return sorted(results, key=lambda item: (item.likes or 0, item.score), reverse=True)
    if request.sort == "downloads":
        return sorted(results, key=lambda item: (item.downloads or 0, item.score), reverse=True)
    return sorted(results, key=lambda item: item.score, reverse=True)


def _hf_dataset_sort(sort: str) -> HfDatasetSort:
    mapping: dict[str, HfDatasetSort] = {
        "createdAt": "created_at",
        "downloads": "downloads",
        "lastModified": "last_modified",
        "likes": "likes",
        "trendingScore": "trending_score",
    }
    return mapping.get(sort, "downloads")


def _stringify_datetime(value: Any) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value)


def _detect_dataset_format(tags: list[str], files: list[str], repo_id: str) -> str | None:
    lower_tags = " ".join(tags).lower()
    lower_files = " ".join(files).lower()
    repo_lower = repo_id.lower()
    if "lerobot" in lower_tags or "meta/info.json" in lower_files or "lerobot" in repo_lower:
        return "lerobot-v3" if "meta/info.json" in lower_files else "lerobot"
    if ".tfrecord" in lower_files or "rlds" in lower_tags:
        return "rlds"
    if ".hdf5" in lower_files or ".h5" in lower_files or "hdf5" in lower_tags:
        return "hdf5"
    if ".zarr" in lower_files or "zarr" in lower_tags:
        return "zarr"
    if ".mcap" in lower_files or "mcap" in lower_tags:
        return "mcap"
    if ".parquet" in lower_files:
        return "parquet"
    return None


def _extract_robots(values: list[str]) -> list[str]:
    known = {
        "franka": ["franka"],
        "aloha": ["aloha"],
        "so100": ["so100", "so-100", "so_arm100"],
        "so101": ["so101", "so-101", "so_arm101"],
        "widowx": ["widowx", "widow-x"],
        "jaco": ["jaco"],
        "sawyer": ["sawyer"],
        "ur5": ["ur5"],
        "kuka": ["kuka"],
        "xarm": ["xarm"],
        "lekiwi": ["lekiwi"],
        "reachy": ["reachy"],
        "unitree": ["unitree"],
        "rebot": ["rebot"],
        "openarm": ["openarm", "open_arm"],
        "koch": ["koch"],
        "omx": ["omx"],
    }
    text = _normalize_robot_text(" ".join(values))
    return [
        robot
        for robot, aliases in known.items()
        if any(_normalize_robot_text(alias) in text for alias in aliases)
    ]


def _normalize_robot_text(value: str) -> str:
    return "".join(char for char in value.lower() if char.isalnum())


def _first_hf_source(sources: Any) -> str | None:
    if not isinstance(sources, list):
        return None
    for source in sources:
        if isinstance(source, dict) and source.get("type") == "hf_hub":
            return str(source.get("uri"))
    return None


def _conversion_hint(fmt: str | None, dataset_ref: str) -> str | None:
    if not fmt:
        return None
    source = dataset_ref if "/" not in dataset_ref else f"hf://{dataset_ref}"
    if "lerobot" in fmt:
        return f"Already LeRobot-like. Use `lerobot-inspect_dataset_metadata` or `forge inspect {source}`."
    return f"Use Forge: `forge convert {source} ./output --format lerobot-v3`."
