#!/usr/bin/env python3
"""
Evaluate GUI generation with GPT-4o.

Usage:
    python scripts/evaluate.py --data-type type1 --output-folder outputs/gemini/01_single_step --dataset-root data
    python scripts/evaluate.py --data-type type2 --output-folder outputs/gemini/02_multi_step --dataset-root data
    python scripts/evaluate.py --data-type type5 --output-folder outputs/gemini/05_grounding_data --dataset-root data
"""

import argparse
import os
from pathlib import Path
from gui_agent.api import Evaluator
from gui_agent.config import EvaluationConfig


def main():
    parser = argparse.ArgumentParser(description="Evaluate GUI generation with GPT-4o")
    parser.add_argument(
        "--data-type",
        required=True,
        choices=["type1", "type2", "type3", "type4", "type5"],
        help="Data type to evaluate"
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        type=Path,
        help="Output folder containing generated images"
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Root dataset folder (for metadata)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers"
    )

    args = parser.parse_args()

    # Validate input
    if not args.api_key:
        print("ERROR: OpenAI API key required")
        print("Set OPENAI_API_KEY environment variable or use --api-key")
        return 1

    if not args.output_folder.exists():
        print(f"ERROR: Output folder not found: {args.output_folder}")
        return 1

    if not args.dataset_root.exists():
        print(f"ERROR: Dataset root not found: {args.dataset_root}")
        return 1

    # Create config and evaluator
    config = EvaluationConfig(
        judge="gpt4o",
        api_key=args.api_key,
        dataset_root=args.dataset_root,
    )

    evaluator = Evaluator(config)

    # Evaluate
    print(f"Evaluating {args.data_type} with GPT-4o...")
    print(f"Output folder: {args.output_folder.absolute()}")
    print(f"Dataset root:  {args.dataset_root.absolute()}")
    print(f"Workers: {args.workers}")
    print()

    results = evaluator.evaluate(
        data_type=args.data_type,
        output_folder=args.output_folder,
        workers=args.workers,
    )

    # Print summary
    if results:
        avg_score = sum(r.overall for r in results if r.overall) / len([r for r in results if r.overall])
        print(f"\n✓ Evaluation complete!")
        print(f"  Evaluated: {len(results)} samples")
        print(f"  Average score: {avg_score:.2f}")
    else:
        print("\n⚠ No results to evaluate")

    return 0


if __name__ == "__main__":
    exit(main())
