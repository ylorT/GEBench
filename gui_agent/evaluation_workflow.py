"""Evaluation workflow orchestrator."""

from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Iterator
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .schemas import EvaluationRequest, EvaluationResult
from .evaluation.registry import create_judge


logger = logging.getLogger(__name__)


class EvaluationWorkflow:
    """
    High-level evaluation workflow manager.

    Handles:
    - Judge setup and validation
    - Sample processing with progress tracking
    - Error handling and result aggregation
    - Parallel execution

    Example:
        workflow = EvaluationWorkflow(
            EvaluationRequest(
                judge="gpt4o",
                api_key="xxx",
                data_type="type1",
                output_folder=Path("outputs")
            )
        )
        results = workflow.evaluate_folder(Path("outputs/gemini/01_single_step"))
    """

    def __init__(self, request: EvaluationRequest):
        """Initialize workflow with request configuration."""
        self.request = request
        self.judge = None
        self._setup()

    def _setup(self) -> None:
        """Setup judge and validate configuration."""
        logger.info(f"Setting up {self.request.judge} judge for {self.request.data_type}")
        self.judge = create_judge(
            data_type=self.request.data_type,
            judge_name=self.request.judge,
            api_key=self.request.api_key,
        )
        if not self.judge:
            raise ValueError(f"Failed to create judge: {self.request.judge}")

    @contextmanager
    def task_context(self) -> Iterator:
        """
        Context manager for task execution.

        Handles setup and cleanup for each evaluation task.
        """
        try:
            yield
        except Exception as e:
            logger.error(f"Error in task context: {e}")
            raise

    def evaluate_sample(self, sample_path: Path) -> EvaluationResult:
        """
        Evaluate a single sample.

        Args:
            sample_path: Path to sample folder

        Returns:
            EvaluationResult with scores
        """
        try:
            with self.task_context():
                result = self.judge.evaluate(sample_path)
                return result
        except Exception as e:
            logger.error(f"Failed to evaluate {sample_path}: {e}")
            return EvaluationResult(
                sample_path=sample_path,
                data_type=self.request.data_type,
                evaluator_model=self.request.judge,
                error_message=str(e)
            )

    def evaluate_batch(
        self,
        samples: List[Path],
        desc: str = "Evaluating"
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple samples with progress bar.

        Args:
            samples: List of sample paths
            desc: Progress bar description

        Returns:
            List of EvaluationResults
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.request.workers) as executor:
            futures = {
                executor.submit(self.evaluate_sample, sample): sample
                for sample in samples
            }

            with tqdm(total=len(samples), desc=desc) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    pbar.set_postfix({"score": f"{result.overall_score:.2f}"})

        return results

    def evaluate_folder(
        self,
        output_folder: Path,
        pattern: str = "folder_*"
    ) -> List[EvaluationResult]:
        """
        Evaluate all samples in a folder.

        Args:
            output_folder: Folder containing generated outputs
            pattern: Sample folder pattern (default: "folder_*")

        Returns:
            List of EvaluationResults
        """
        # Discover all samples
        samples = sorted(output_folder.glob(pattern))

        if not samples:
            logger.warning(f"No samples found in {output_folder}")
            return []

        logger.info(f"Found {len(samples)} samples for evaluation")

        # Evaluate batch
        results = self.evaluate_batch(
            samples,
            desc=f"Evaluating with {self.request.judge}"
        )

        return results

    def get_summary(self, results: List[EvaluationResult]) -> dict:
        """
        Get summary statistics from evaluation results.

        Args:
            results: List of EvaluationResults

        Returns:
            Summary dict with statistics
        """
        valid_results = [r for r in results if r.overall_score > 0]
        scores = [r.overall_score for r in valid_results]

        if not scores:
            return {
                "total": len(results),
                "evaluated": 0,
                "failed": len(results),
                "avg_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
            }

        return {
            "total": len(results),
            "evaluated": len(valid_results),
            "failed": len(results) - len(valid_results),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
        }
