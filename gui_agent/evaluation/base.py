"""Base classes and interfaces for evaluation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List
import json
from datetime import datetime


@dataclass
class EvaluationResult:
    """Result of a single sample evaluation."""

    sample_name: str
    data_type: str
    evaluator_model: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Scores for different dimensions (0-5 scale)
    scores: Dict[str, int] = field(default_factory=dict)

    # Overall normalized score (0-1 scale)
    overall: Optional[float] = None

    # Justification for the evaluation
    justification: str = ""

    # Raw response from evaluator
    raw_response: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "sample_name": self.sample_name,
            "data_type": self.data_type,
            "evaluator_model": self.evaluator_model,
            "timestamp": self.timestamp,
            "scores": self.scores,
            "overall": self.overall,
            "justification": self.justification,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class BaseJudgeProvider(ABC):
    """Abstract base class for evaluation providers (API or local)."""

    def __init__(self, api_key: Optional[str] = None, api_endpoint: Optional[str] = None):
        self.api_key = api_key
        self.api_endpoint = api_endpoint

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        pass

    @abstractmethod
    def evaluate(
        self,
        sample_data: Dict,
        **kwargs
    ) -> Dict[str, any]:
        """
        Evaluate a sample and return scores.

        Args:
            sample_data: Dictionary containing evaluation context and images
            **kwargs: Additional provider-specific parameters

        Returns:
            Dictionary with keys like 'goal', 'logic', 'consistency', 'ui', 'quality'
            Each value should be an integer 0-5 with optional justification
        """
        pass

    def validate_config(self) -> bool:
        """Check if provider is properly configured."""
        return self.api_key is not None


class BaseJudge(ABC):
    """Abstract base class for data type specific judges."""

    def __init__(self, judge_provider: BaseJudgeProvider, dataset_root: Optional[Path] = None):
        self.judge_provider = judge_provider
        self.dataset_root = dataset_root

    @property
    @abstractmethod
    def data_type(self) -> str:
        """Data type identifier (e.g., 'type1', 'type2')."""
        pass

    @abstractmethod
    def evaluate_sample(self, sample_path: Path) -> Optional[EvaluationResult]:
        """
        Evaluate a single sample folder.

        Args:
            sample_path: Path to sample folder containing generated outputs

        Returns:
            EvaluationResult if successful, None otherwise
        """
        pass

    def _load_metadata(self, meta_path: Path) -> dict:
        """Load metadata JSON."""
        if not meta_path.exists():
            return {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _load_image(self, image_path: Path):
        """Load image from file."""
        try:
            if image_path.exists():
                from PIL import Image
                return Image.open(image_path)
        except Exception:
            pass
        return None

    def _find_image(self, folder: Path, suffix: Optional[str] = None):
        """Find image file in folder."""
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
            if suffix:
                candidates = sorted(folder.glob(f"*{suffix}{ext}"))
            else:
                candidates = sorted(folder.glob(f"*{ext}"))
            if candidates:
                return candidates[0]
        return None
