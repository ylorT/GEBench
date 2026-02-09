"""Pydantic schemas for data validation."""

from pathlib import Path
from typing import Optional, Dict, List
from pydantic import BaseModel, Field, field_validator


class SampleMetadata(BaseModel):
    """Sample metadata structure."""
    caption: Optional[str] = None
    lang_device: str = Field(default="english_phone")
    image_path: Optional[str] = None
    goal: Optional[str] = None
    trajectory: Optional[List[Dict]] = None
    location_info: Optional[Dict] = None

    class Config:
        extra = "allow"  # Allow extra fields


class GenerationRequest(BaseModel):
    """Generation request configuration."""
    provider: str = Field(default="gemini")
    api_key: str
    output_dir: Path
    data_type: str = Field(default="type1")
    workers: int = Field(default=4, ge=1, le=32)
    max_retries: int = Field(default=3, ge=1, le=10)
    timeout: int = Field(default=300, ge=10)
    batch_size: int = Field(default=1, ge=1)

    @field_validator("output_dir", mode="before")
    @classmethod
    def validate_output_dir(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v


class EvaluationRequest(BaseModel):
    """Evaluation request configuration."""
    judge: str = Field(default="gpt4o")
    api_key: str
    data_type: str = Field(default="type1")
    dataset_root: Path
    output_folder: Path
    workers: int = Field(default=4, ge=1, le=32)
    timeout: int = Field(default=300, ge=10)

    @field_validator("dataset_root", "output_folder", mode="before")
    @classmethod
    def validate_paths(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v


class GenerationResult(BaseModel):
    """Result of a single generation task."""
    sample_path: Path
    output_path: Path
    status: str = "success"  # success, failed, skipped
    error_message: Optional[str] = None


class EvaluationResult(BaseModel):
    """Evaluation result with scores."""
    sample_path: Path
    data_type: str
    evaluator_model: str
    scores: Dict[str, float] = Field(default_factory=dict)
    overall_score: float = 0.0
    justification: str = ""
    timestamp: Optional[str] = None
