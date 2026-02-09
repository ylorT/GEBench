"""High-level Generator and Evaluator API."""

from pathlib import Path
from typing import Optional, List, Iterable
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import GenerationConfig, EvaluationConfig
from .generation.providers import get_provider
from .generation.registry import create_generator
from .evaluation.providers import get_judge_provider
from .evaluation.type1 import Type1Judge
from .evaluation.type2 import Type2Judge
from .evaluation.type3 import Type3Judge
from .evaluation.type4 import Type4Judge
from .evaluation.type5 import Type5Judge


class Generator:
    """High-level API for GUI generation."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.provider = get_provider(
            config.provider,
            api_key=config.api_key,
            api_endpoint=config.api_endpoint,
        )

    @classmethod
    def from_env(cls, provider: str = "gemini") -> "Generator":
        """Create Generator from environment variables."""
        config = GenerationConfig.from_env(provider)
        return cls(config)

    def generate(self, data_type: str, data_folder: Path, workers: int = 1) -> None:
        """
        Generate GUIs for a data type.

        Args:
            data_type: Type identifier ('type1', 'type2', 'type3', 'type4', 'type5')
            data_folder: Root folder containing sample folders
            workers: Number of parallel workers
        """
        generator = create_generator(
            data_type,
            self.provider,
            self.config.output_dir,
            dataset_root=self.config.dataset_root,
        )

        # Collect samples
        samples = list(self._iter_samples(data_folder, data_type))

        if not samples:
            print(f"No samples found in {data_folder}")
            return

        print(f"Processing {len(samples)} samples with {workers} workers...")

        if workers == 1:
            for sample in samples:
                generator.process_sample(sample)
        else:
            with Pool(processes=workers) as pool:
                for _ in pool.imap_unordered(generator.process_sample, samples):
                    pass

    @staticmethod
    def _iter_samples(folder: Path, data_type: str) -> Iterable[Path]:
        """Iterate over sample folders."""
        if not folder.exists():
            return

        if data_type in ["type1", "type2", "type5"]:
            # These types have lang_device subdirectories
            for lang_dir in sorted(folder.iterdir()):
                if lang_dir.is_dir():
                    for sample_dir in sorted(lang_dir.iterdir()):
                        if sample_dir.is_dir() and sample_dir.name.startswith("folder_"):
                            yield sample_dir
        elif data_type in ["type3", "type4"]:
            # These types have JSON files
            for lang_dir in sorted(folder.iterdir()):
                if lang_dir.is_dir():
                    for json_file in sorted(lang_dir.glob("*.json")):
                        yield json_file


class Evaluator:
    """High-level API for GUI evaluation."""

    JUDGE_CLASSES = {
        "type1": Type1Judge,
        "type2": Type2Judge,
        "type3": Type3Judge,
        "type4": Type4Judge,
        "type5": Type5Judge,
    }

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.judge_provider = get_judge_provider(
            config.judge,
            api_key=config.api_key,
            api_endpoint=config.api_endpoint,
        )

    @classmethod
    def from_env(cls, judge: str = "gpt4o") -> "Evaluator":
        """Create Evaluator from environment variables."""
        config = EvaluationConfig.from_env(judge)
        return cls(config)

    def evaluate(self, data_type: str, output_folder: Path, workers: int = 1) -> List:
        """
        Evaluate generated GUIs.

        Args:
            data_type: Type identifier ('type1', 'type2', 'type3', 'type4', 'type5')
            output_folder: Folder containing generated outputs
            workers: Number of parallel workers

        Returns:
            List of EvaluationResult objects
        """
        if data_type not in self.JUDGE_CLASSES:
            raise ValueError(f"Evaluation not supported for {data_type}")

        JudgeClass = self.JUDGE_CLASSES[data_type]
        judge = JudgeClass(self.judge_provider, self.config.dataset_root)

        # Collect samples
        samples = list(self._iter_samples(output_folder))

        if not samples:
            print(f"No samples found in {output_folder}")
            return []

        print(f"Evaluating {len(samples)} samples with {workers} workers...")

        results = []
        if workers == 1:
            for sample in samples:
                result = judge.evaluate_sample(sample)
                if result:
                    results.append(result)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(judge.evaluate_sample, s): s for s in samples}
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)

        # Save results
        self._save_results(results, output_folder)
        return results

    @staticmethod
    def _iter_samples(folder: Path) -> Iterable[Path]:
        """Iterate over sample folders."""
        if not folder.exists():
            return

        for lang_dir in sorted(folder.iterdir()):
            if lang_dir.is_dir():
                for sample_dir in sorted(lang_dir.iterdir()):
                    if sample_dir.is_dir():
                        yield sample_dir

    def _save_results(self, results, output_folder: Path) -> None:
        """Save evaluation results to output folder."""
        import json
        from datetime import datetime

        results_file = output_folder.parent / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        data = {
            "evaluator": self.config.judge,
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in results],
        }

        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {results_file}")
