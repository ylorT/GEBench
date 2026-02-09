"""Configuration management for GUI Agent."""

import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class GenerationConfig:
    """Configuration for generation operations."""

    # Provider and API settings
    provider: str = "gemini"  # "gemini", "gpt", "seedream"
    api_key: Optional[str] = None
    api_endpoint: Optional[str] = None

    # Generation settings
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    batch_size: int = 1
    max_workers: int = 4
    max_retries: int = 3
    timeout: int = 300

    # Data settings
    dataset_root: Optional[Path] = None
    data_types: list = field(default_factory=lambda: [1, 2, 3, 4, 5])

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if self.dataset_root and isinstance(self.dataset_root, str):
            self.dataset_root = Path(self.dataset_root)

        # Load API key from environment if not provided
        if not self.api_key:
            env_var = f"GUI_AGENT_API_KEY_{self.provider.upper()}"
            self.api_key = os.getenv(env_var)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, provider: str = "gemini") -> "GenerationConfig":
        """Create config from environment variables."""
        return cls(
            provider=provider,
            api_key=os.getenv(f"GUI_AGENT_API_KEY_{provider.upper()}"),
            output_dir=Path(os.getenv("GUI_AGENT_OUTPUT_DIR", "outputs")),
        )

    @classmethod
    def from_file(cls, path: str) -> "GenerationConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_file(self, path: str) -> None:
        """Save config to JSON file."""
        data = {
            "provider": self.provider,
            "output_dir": str(self.output_dir),
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
        }
        if self.dataset_root:
            data["dataset_root"] = str(self.dataset_root)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation operations."""

    # Judge and API settings
    judge: str = "gpt4o"  # "gpt4o", "gemini", "qwen_vl"
    api_key: Optional[str] = None
    api_endpoint: Optional[str] = None

    # Evaluation settings
    output_format: str = "json"  # "json", "csv"
    max_workers: int = 4
    max_retries: int = 3
    timeout: int = 300

    # Data settings
    dataset_root: Optional[Path] = None
    eval_types: list = field(default_factory=lambda: [1, 2, 5])

    def __post_init__(self):
        if self.dataset_root and isinstance(self.dataset_root, str):
            self.dataset_root = Path(self.dataset_root)

        # Load API key from environment if not provided
        if not self.api_key:
            env_var = f"GUI_AGENT_API_KEY_{self.judge.upper()}"
            self.api_key = os.getenv(env_var)

    @classmethod
    def from_env(cls, judge: str = "gpt4o") -> "EvaluationConfig":
        """Create config from environment variables."""
        return cls(
            judge=judge,
            api_key=os.getenv(f"GUI_AGENT_API_KEY_{judge.upper()}"),
        )

    @classmethod
    def from_file(cls, path: str) -> "EvaluationConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_file(self, path: str) -> None:
        """Save config to JSON file."""
        data = {
            "judge": self.judge,
            "output_format": self.output_format,
            "max_workers": self.max_workers,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
        }
        if self.dataset_root:
            data["dataset_root"] = str(self.dataset_root)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


@dataclass
class Config:
    """Complete configuration for GUI Agent."""

    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Load complete config from file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            generation=GenerationConfig(**data.get("generation", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
        )

    def to_file(self, path: str) -> None:
        """Save complete config to file."""
        data = {
            "generation": {
                "provider": self.generation.provider,
                "output_dir": str(self.generation.output_dir),
            },
            "evaluation": {
                "judge": self.evaluation.judge,
                "output_format": self.evaluation.output_format,
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
