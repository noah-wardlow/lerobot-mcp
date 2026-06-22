from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

type PrimitiveValue = str | int | float | bool | None
type JsonValue = PrimitiveValue | list[JsonValue] | dict[str, JsonValue]
OPTION_KEY_PATTERN = r"^[A-Za-z0-9_.-]+$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RepoType(StrEnum):
    DATASET = "dataset"
    MODEL = "model"
    SPACE = "space"


class JobState(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CommandOption(StrictModel):
    key: str = Field(min_length=1, pattern=OPTION_KEY_PATTERN)
    value: PrimitiveValue = None


def validate_option_keys(options: dict[str, PrimitiveValue]) -> dict[str, PrimitiveValue]:
    import re

    for key in options:
        if not re.fullmatch(OPTION_KEY_PATTERN, key):
            raise ValueError(f"Invalid option key: {key!r}")
    return options


class CommandRequest(StrictModel):
    command: str = Field(min_length=1)
    options: dict[str, PrimitiveValue] = Field(default_factory=dict)
    extra_args: list[str] = Field(default_factory=list)
    cwd: Path | None = None
    timeout_seconds: int = Field(default=900, ge=1, le=86_400)
    background: bool = True
    use_uv: bool = True
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("extra_args")
    @classmethod
    def validate_extra_args(cls, value: list[str]) -> list[str]:
        if any(arg == "" or "\x00" in arg for arg in value):
            raise ValueError("extra_args cannot contain empty strings or NUL bytes")
        return value

    @field_validator("options")
    @classmethod
    def validate_options(cls, value: dict[str, PrimitiveValue]) -> dict[str, PrimitiveValue]:
        return validate_option_keys(value)


class ExampleRequest(StrictModel):
    example_path: str = Field(min_length=1)
    options: dict[str, PrimitiveValue] = Field(default_factory=dict)
    extra_args: list[str] = Field(default_factory=list)
    timeout_seconds: int = Field(default=900, ge=1, le=86_400)
    background: bool = True
    use_uv: bool = True
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("extra_args")
    @classmethod
    def validate_extra_args(cls, value: list[str]) -> list[str]:
        if any(arg == "" or "\x00" in arg for arg in value):
            raise ValueError("extra_args cannot contain empty strings or NUL bytes")
        return value

    @field_validator("options")
    @classmethod
    def validate_options(cls, value: dict[str, PrimitiveValue]) -> dict[str, PrimitiveValue]:
        return validate_option_keys(value)


class CommandPreview(StrictModel):
    argv: list[str]
    cwd: Path | None
    env: dict[str, str]


class ProcessResult(StrictModel):
    argv: list[str]
    cwd: Path | None
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


class JobInfo(StrictModel):
    job_id: str
    state: JobState
    argv: list[str]
    cwd: Path | None
    returncode: int | None
    started_at: float
    finished_at: float | None
    stdout_tail: str
    stderr_tail: str


class ExampleInfo(StrictModel):
    path: str
    category: str
    name: str


class DatasetMetadataRequest(StrictModel):
    repo_id: str = Field(min_length=1)
    root: Path | None = None
    revision: str | None = None
    force_cache_sync: bool = False
    timeout_seconds: int = Field(default=120, ge=1, le=3_600)
    use_uv: bool = True


class DatasetLatestFormatConvertRequest(StrictModel):
    repo_id: str = Field(min_length=1)
    root: Path | None = None
    branch: str | None = None
    target_version: Literal["v3.0"] = "v3.0"
    data_file_size_in_mb: int | None = Field(default=None, ge=1)
    video_file_size_in_mb: int | None = Field(default=None, ge=1)
    push_to_hub: bool = False
    force_conversion: bool = False
    timeout_seconds: int = Field(default=86_400, ge=1, le=604_800)
    background: bool = True
    use_uv: bool = True
    env: dict[str, str] = Field(default_factory=dict)


class HubRepoInfo(StrictModel):
    repo_id: str
    repo_type: RepoType
    private: bool | None = None
    sha: str | None = None
    tags: list[str] = Field(default_factory=list)
    siblings: list[str] = Field(default_factory=list)


class LeRobotCommandInfo(StrictModel):
    name: str
    target: str
    module: str
    function: str
    shorthand: str


class RegistryItem(StrictModel):
    name: str
    registry: str
    class_name: str | None = None
    module_path: str | None = None
    source_path: str


class LeRobotCapabilities(StrictModel):
    git_commit: str | None
    commands: list[LeRobotCommandInfo]
    extras: dict[str, list[str]]
    examples: list[ExampleInfo]
    registries: dict[str, list[RegistryItem]]


class ForgeConvertRequest(StrictModel):
    source: str = Field(min_length=1)
    output: Path
    target_format: str = "lerobot-v3"
    source_format: str | None = None
    config_file: Path | None = None
    fps: float | None = Field(default=None, gt=0)
    robot_type: str | None = None
    camera_mapping: dict[str, str] = Field(default_factory=dict)
    workers: int = Field(default=1, ge=1, le=256)
    fail_on_error: bool = False
    visualize: bool = False
    dry_run: bool = False
    background: bool = True
    timeout_seconds: int = Field(default=3_600, ge=1, le=604_800)
    env: dict[str, str] = Field(default_factory=dict)


class ForgeInspectRequest(StrictModel):
    path: str = Field(min_length=1)
    format: str | None = None
    output: Literal["text", "json"] = "json"
    quick: bool = True
    deep: bool = False
    samples: int = Field(default=5, ge=1, le=1_000)
    background: bool = False
    timeout_seconds: int = Field(default=600, ge=1, le=86_400)
    env: dict[str, str] = Field(default_factory=dict)


class DatasetSearchRequest(StrictModel):
    query: str | None = None
    robot: str | None = None
    format: str | None = None
    min_episodes: int | None = Field(default=None, ge=0)
    max_episodes: int | None = Field(default=None, ge=0)
    max_size_gb: float | None = Field(default=None, gt=0)
    tags: list[str] = Field(default_factory=list)
    task: str | None = None
    language_conditioned: bool | None = None
    simulation: bool | None = None
    demo_suitable: bool | None = None
    prefer_lerobot: bool = True
    include_forge_registry: bool = True
    include_hub: bool = True
    sort: Literal["downloads", "likes", "lastModified", "createdAt", "trendingScore"] = "downloads"
    limit: int = Field(default=20, ge=1, le=100)


class DatasetSearchResult(StrictModel):
    id: str
    name: str | None = None
    source: Literal["hf", "forge_registry"]
    repo_id: str | None = None
    format: str | None = None
    robot: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    task_types: list[str] = Field(default_factory=list)
    episodes: int | None = None
    hours: float | None = None
    size_bytes: int | None = None
    downloads: int | None = None
    likes: int | None = None
    private: bool | None = None
    created_at: str | None = None
    last_modified: str | None = None
    score: float
    conversion_hint: str | None = None
    notes: str | None = None


class PolicyRepoInspectRequest(StrictModel):
    repo_id: str = Field(min_length=1)
    revision: str | None = None
    include_raw_configs: bool = False


class PolicyRepoInspection(StrictModel):
    repo_id: str
    revision: str | None = None
    sha: str | None = None
    private: bool | None = None
    tags: list[str] = Field(default_factory=list)
    config_files: list[str] = Field(default_factory=list)
    weight_files: list[str] = Field(default_factory=list)
    processor_files: list[str] = Field(default_factory=list)
    policy_type: str | None = None
    dataset_repo_id: str | None = None
    robot_type: str | None = None
    fps: float | None = None
    input_features: dict[str, JsonValue] = Field(default_factory=dict)
    output_features: dict[str, JsonValue] = Field(default_factory=dict)
    image_keys: list[str] = Field(default_factory=list)
    state_keys: list[str] = Field(default_factory=list)
    action_keys: list[str] = Field(default_factory=list)
    missing_expected_files: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    raw_configs: dict[str, JsonValue] = Field(default_factory=dict)


type CommandMode = Literal["foreground", "background"]
