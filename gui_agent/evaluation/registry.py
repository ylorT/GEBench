"""Judge registry and factory for creating judges."""

from typing import Optional

from .base import BaseJudge
from .type1 import Type1Judge
from .type2 import Type2Judge
from .type3 import Type3Judge
from .type4 import Type4Judge
from .type5 import Type5Judge
from .providers import get_judge_provider


JUDGE_MAP = {
    "type1": Type1Judge,
    "type2": Type2Judge,
    "type3": Type3Judge,
    "type4": Type4Judge,
    "type5": Type5Judge,
}


def create_judge(
    data_type: str,
    judge_name: str,
    api_key: str,
) -> Optional[BaseJudge]:
    """
    Create a judge instance.

    Args:
        data_type: Data type ('type1', 'type2', 'type5')
        judge_name: Judge name ('gpt4o', 'gemini', 'qwen_vl')
        api_key: API key for judge

    Returns:
        Judge instance or None if creation failed
    """
    if data_type not in JUDGE_MAP:
        raise ValueError(
            f"Unknown data type: {data_type}. "
            f"Available: {list(JUDGE_MAP.keys())}"
        )

    try:
        # Create judge provider
        provider = get_judge_provider(
            name=judge_name,
            api_key=api_key
        )

        # Create judge
        judge_class = JUDGE_MAP[data_type]
        judge = judge_class(provider=provider)

        return judge
    except Exception as e:
        raise RuntimeError(f"Failed to create judge: {e}")


def list_judges() -> list:
    """Get list of available judge types."""
    return list(JUDGE_MAP.keys())
