"""Base classes and interfaces for generation providers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import json


class BaseProvider(ABC):
    """Abstract base class for generation providers (API or local)."""

    def __init__(self, api_key: Optional[str] = None, api_endpoint: Optional[str] = None):
        self.api_key = api_key
        self.api_endpoint = api_endpoint

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        reference_image: Optional[Image.Image] = None,
        **kwargs
    ) -> Image.Image:
        """
        Generate an image based on prompt and optional reference image.

        Args:
            prompt: Text prompt for generation
            reference_image: Optional reference image for conditioning
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated PIL Image
        """
        pass

    def validate_config(self) -> bool:
        """Check if provider is properly configured."""
        return self.api_key is not None


class PromptBuilder(ABC):
    """Base class for prompt building strategies."""

    @abstractmethod
    def build(self, sample_data: dict) -> str:
        """Build prompt from sample data."""
        pass


class BaseGenerator(ABC):
    """Abstract base class for data type specific generators."""

    def __init__(self, provider: BaseProvider, prompt_builder: PromptBuilder, output_dir: Path):
        self.provider = provider
        self.prompt_builder = prompt_builder
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def data_type(self) -> str:
        """Data type identifier (e.g., 'type1', 'type2')."""
        pass

    @abstractmethod
    def process_sample(self, sample_path: Path) -> Optional[Path]:
        """
        Process a single sample folder and generate output.

        Args:
            sample_path: Path to sample folder

        Returns:
            Path to generated output if successful, None otherwise
        """
        pass

    def _load_metadata(self, meta_path: Path) -> dict:
        """Load and parse metadata JSON."""
        if not meta_path.exists():
            return {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _load_image(self, image_path: Path) -> Optional[Image.Image]:
        """Load image from file."""
        try:
            if image_path.exists():
                return Image.open(image_path)
        except Exception:
            pass
        return None

    def _save_output(self, output_path: Path, image: Image.Image) -> None:
        """Save generated image to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

    def _should_skip(self, output_path: Path, min_size: int = 1024) -> bool:
        """Check if output already exists and is valid."""
        return output_path.exists() and output_path.stat().st_size > min_size
