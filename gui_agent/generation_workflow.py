"""Generation workflow orchestrator."""

from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Iterator
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .schemas import GenerationRequest, GenerationResult, SampleMetadata
from .generation.base import BaseGenerator, BaseProvider
from .generation.registry import create_generator


logger = logging.getLogger(__name__)


class GenerationWorkflow:
    """
    High-level generation workflow manager.

    Handles:
    - Task submission and result collection
    - Progress tracking with tqdm
    - Error handling and retries
    - Parallel execution

    Example:
        workflow = GenerationWorkflow(
            GenerationRequest(
                provider="gemini",
                api_key="xxx",
                output_dir=Path("outputs"),
                data_type="type1"
            )
        )
        results = workflow.process_samples(samples)
    """

    def __init__(self, request: GenerationRequest):
        """Initialize workflow with request configuration."""
        self.request = request
        self.generator: Optional[BaseGenerator] = None
        self._setup()

    def _setup(self) -> None:
        """Setup generator and validate configuration."""
        logger.info(f"Setting up {self.request.data_type} generator with {self.request.provider} provider")
        self.generator = create_generator(
            data_type=self.request.data_type,
            provider_name=self.request.provider,
            api_key=self.request.api_key,
            output_dir=self.request.output_dir,
        )
        if not self.generator:
            raise ValueError(f"Failed to create generator for {self.request.data_type}")

    @contextmanager
    def task_context(self) -> Iterator:
        """
        Context manager for task execution.

        Handles setup and cleanup for each task.
        """
        try:
            yield
        except Exception as e:
            logger.error(f"Error in task context: {e}")
            raise

    def process_sample(self, sample_path: Path) -> GenerationResult:
        """
        Process a single sample.

        Args:
            sample_path: Path to sample folder

        Returns:
            GenerationResult with status and output path
        """
        try:
            with self.task_context():
                output_path = self.generator.process_sample(sample_path)
                if output_path:
                    return GenerationResult(
                        sample_path=sample_path,
                        output_path=output_path,
                        status="success"
                    )
                else:
                    return GenerationResult(
                        sample_path=sample_path,
                        output_path=sample_path,
                        status="skipped"
                    )
        except Exception as e:
            logger.error(f"Failed to process {sample_path}: {e}")
            return GenerationResult(
                sample_path=sample_path,
                output_path=sample_path,
                status="failed",
                error_message=str(e)
            )

    def process_batch(
        self,
        samples: List[Path],
        desc: str = "Generating"
    ) -> List[GenerationResult]:
        """
        Process multiple samples with progress bar.

        Args:
            samples: List of sample paths
            desc: Progress bar description

        Returns:
            List of GenerationResults
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.request.workers) as executor:
            futures = {
                executor.submit(self.process_sample, sample): sample
                for sample in samples
            }

            with tqdm(total=len(samples), desc=desc) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)

                    # Update progress description with status
                    status_icon = "✓" if result.status == "success" else (
                        "⊘" if result.status == "skipped" else "✗"
                    )
                    pbar.set_postfix({"status": f"{status_icon} {result.status}"})

        return results

    def process_folder(
        self,
        data_folder: Path,
        pattern: str = "folder_*"
    ) -> List[GenerationResult]:
        """
        Process all samples in a folder.

        Args:
            data_folder: Root data folder containing samples
            pattern: Sample folder pattern (default: "folder_*")

        Returns:
            List of GenerationResults
        """
        # Discover all samples
        samples = sorted(data_folder.glob(pattern))

        if not samples:
            logger.warning(f"No samples found in {data_folder} with pattern {pattern}")
            return []

        logger.info(f"Found {len(samples)} samples in {data_folder}")

        # Process batch
        results = self.process_batch(
            samples,
            desc=f"Processing {self.request.data_type}"
        )

        return results

    def get_summary(self, results: List[GenerationResult]) -> dict:
        """
        Get summary statistics from results.

        Args:
            results: List of GenerationResults

        Returns:
            Summary dict with counts and success rate
        """
        total = len(results)
        success = sum(1 for r in results if r.status == "success")
        failed = sum(1 for r in results if r.status == "failed")
        skipped = sum(1 for r in results if r.status == "skipped")

        return {
            "total": total,
            "success": success,
            "failed": failed,
            "skipped": skipped,
            "success_rate": success / total if total > 0 else 0,
        }
