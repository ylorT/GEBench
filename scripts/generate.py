#!/usr/bin/env python3
"""
Generate GUI images using Gemini.

Usage:
    python scripts/generate.py --data-type type1 --data-folder data/01_single_step --output-dir outputs/gemini
    python scripts/generate.py --data-type type2 --data-folder data/02_multi_step --output-dir outputs/gemini
    python scripts/generate.py --data-type type5 --data-folder data/05_grounding_data --output-dir outputs/gemini
"""

import argparse
import os
from pathlib import Path
from gui_agent.api import Generator
from gui_agent.config import GenerationConfig


def main():
    parser = argparse.ArgumentParser(description="Generate GUI images using Gemini")
    parser.add_argument(
        "--data-type",
        required=True,
        choices=["type1", "type2", "type3", "type4", "type5"],
        help="Data type to generate"
    )
    parser.add_argument(
        "--data-folder",
        required=True,
        type=Path,
        help="Input data folder"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for generated images"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("GEMINI_API_KEY"),
        help="Gemini API key (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Root dataset folder (optional, for metadata)"
    )

    args = parser.parse_args()

    # Validate input
    if not args.api_key:
        print("ERROR: Gemini API key required")
        print("Set GEMINI_API_KEY environment variable or use --api-key")
        return 1

    if not args.data_folder.exists():
        print(f"ERROR: Data folder not found: {args.data_folder}")
        return 1

    # Create config and generator
    config = GenerationConfig(
        provider="gemini",
        api_key=args.api_key,
        output_dir=args.output_dir,
        dataset_root=args.dataset_root,
    )

    generator = Generator(config)

    # Generate
    print(f"Generating {args.data_type} images...")
    print(f"Input:  {args.data_folder.absolute()}")
    print(f"Output: {args.output_dir.absolute()}")
    print(f"Workers: {args.workers}")
    print()

    generator.generate(
        data_type=args.data_type,
        data_folder=args.data_folder,
        workers=args.workers,
    )

    print("\n✓ Generation complete!")
    return 0


if __name__ == "__main__":
    exit(main())
