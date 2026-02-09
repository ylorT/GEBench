"""Generator registry and factory."""

from pathlib import Path
from typing import Optional, Dict, Type

from .base import BaseGenerator, BaseProvider
from .type1 import Type1Generator
from .type2 import Type2Generator
from .type34 import Type3Generator, Type4Generator
from .type5 import Type5Generator


GENERATOR_MAP: Dict[str, Type[BaseGenerator]] = {
    "type1": Type1Generator,
    "type2": Type2Generator,
    "type3": Type3Generator,
    "type4": Type4Generator,
    "type5": Type5Generator,
}


def create_generator(
    data_type: str,
    provider: BaseProvider,
    output_dir: Path,
    dataset_root: Optional[Path] = None,
) -> BaseGenerator:
    """
    Create a generator instance for the specified data type.

    Args:
        data_type: Type identifier ('type1', 'type2', 'type3', 'type4', 'type5')
        provider: Provider instance for generation
        output_dir: Output directory for generated files
        dataset_root: Root directory for dataset metadata

    Returns:
        Appropriate generator instance

    Raises:
        ValueError: If data_type is not recognized
    """
    if data_type not in GENERATOR_MAP:
        raise ValueError(f"Unknown data type: {data_type}. Available: {list(GENERATOR_MAP.keys())}")

    GeneratorClass = GENERATOR_MAP[data_type]
    return GeneratorClass(provider, output_dir, dataset_root)


def list_generators() -> list:
    """List all available generator types."""
    return list(GENERATOR_MAP.keys())
