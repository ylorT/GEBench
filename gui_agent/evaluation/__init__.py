"""Evaluation module initialization."""

from .base import BaseJudge, BaseJudgeProvider, EvaluationResult
from .providers import get_judge_provider, GPT4oProvider
from .prompts import TYPE1_EVAL_PROMPT, TYPE2_EVAL_PROMPT, TYPE5_EVAL_PROMPT
from .type1 import Type1Judge
from .type2 import Type2Judge
from .type3 import Type3Judge
from .type4 import Type4Judge
from .type5 import Type5Judge

__all__ = [
    "BaseJudge",
    "BaseJudgeProvider",
    "EvaluationResult",
    "get_judge_provider",
    "GPT4oProvider",
    "Type1Judge",
    "Type2Judge",
    "Type3Judge",
    "Type4Judge",
    "Type5Judge",
    "TYPE1_EVAL_PROMPT",
    "TYPE2_EVAL_PROMPT",
    "TYPE5_EVAL_PROMPT",
]
