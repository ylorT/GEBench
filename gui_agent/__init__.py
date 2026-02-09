"""
GUI Agent - Open-source toolkit for GUI generation and evaluation.

This package provides tools for:
1. Generating GUI screenshots using text-to-image models (Type 1-5 data types)
2. Evaluating generated GUIs using multiple evaluator models

Two API styles:
1. High-level Workflow API (recommended for complex tasks):
   - GenerationWorkflow: Generate with progress tracking
   - EvaluationWorkflow: Evaluate with progress tracking

2. Simple API (for quick usage):
   - Generator: Simple generation interface
   - Evaluator: Simple evaluation interface
"""

__version__ = "0.1.0"

from .config import Config, GenerationConfig, EvaluationConfig
from .api import Generator, Evaluator
from .generation_workflow import GenerationWorkflow
from .evaluation_workflow import EvaluationWorkflow
from .schemas import (
    GenerationRequest,
    EvaluationRequest,
    GenerationResult,
    EvaluationResult,
    SampleMetadata,
)
from .evaluation.prompts import get_eval_prompt

__all__ = [
    # Configuration
    "Config",
    "GenerationConfig",
    "EvaluationConfig",
    # Simple API
    "Generator",
    "Evaluator",
    # Workflow API (ImgEdit-style)
    "GenerationWorkflow",
    "EvaluationWorkflow",
    # Schemas (Pydantic models)
    "GenerationRequest",
    "EvaluationRequest",
    "GenerationResult",
    "EvaluationResult",
    "SampleMetadata",
    # Evaluation utilities
    "get_eval_prompt",
]

