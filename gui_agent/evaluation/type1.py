"""Type 1 Judge: Single-step UI transition evaluation."""

from pathlib import Path
from typing import Optional, Dict

from .base import BaseJudge, EvaluationResult
from .prompts import get_eval_prompt


class Type1Judge(BaseJudge):
    """Judge for Type 1 (single-step UI transitions)."""

    @property
    def data_type(self) -> str:
        return "type1"

    def evaluate_sample(self, sample_path: Path) -> Optional[EvaluationResult]:
        """
        Evaluate a Type 1 sample.

        Expected structure:
        sample_path/
            ├── gemini.png (or other provider)
            └── (initial image should be in dataset metadata)
        """
        folder_name = sample_path.name
        lang_device = sample_path.parent.name

        # Load generated image
        gen_image_path = self._find_image(sample_path)
        if not gen_image_path:
            print(f"[skip] No generated image in {sample_path}")
            return None

        gen_image = self._load_image(gen_image_path)
        if not gen_image:
            return None

        # Load dataset metadata (initial image and caption)
        if not self.dataset_root:
            print(f"[skip] dataset_root required for evaluation")
            return None

        meta_path = self.dataset_root / "01_single_step" / lang_device / folder_name / "meta_data.json"
        metadata = self._load_metadata(meta_path)

        if not metadata:
            print(f"[skip] No metadata at {meta_path}")
            return None

        # Load initial image
        rel_image = metadata.get("image")
        caption = metadata.get("caption", "")

        if not rel_image:
            print(f"[skip] No image reference in metadata")
            return None

        init_image_path = self.dataset_root / "01_single_step" / str(rel_image)
        init_image = self._load_image(init_image_path)

        if not init_image:
            print(f"[skip] Could not load initial image: {init_image_path}")
            return None

        # Evaluate
        try:
            print(f"[eval] {lang_device}/{folder_name}...")

            sample_data = {
                "images": {
                    "initial": init_image,
                    "generated": gen_image,
                },
                "caption": caption,
            }

            # Call evaluator
            prompt = get_eval_prompt("type1", lang_device)
            scores_dict = self.judge_provider.evaluate(
                sample_data,
                prompt=prompt + f"\n\nCaption: {caption}",
                max_retries=3,
            )

            # Parse scores
            scores = self._parse_scores(scores_dict)
            overall = self._compute_overall(scores)

            result = EvaluationResult(
                sample_name=f"{lang_device}/{folder_name}",
                data_type="type1",
                evaluator_model=self.judge_provider.name,
                scores=scores,
                overall=overall,
            )

            print(f"[done] Overall: {overall:.2f}")
            return result

        except Exception as e:
            print(f"[error] Failed to evaluate: {e}")
            return None

    @staticmethod
    def _parse_scores(response_dict: Dict) -> Dict[str, int]:
        """Parse scores from evaluator response."""
        scores = {}
        dimension_names = ["goal", "logic", "cons", "ui", "qual"]

        for dim in dimension_names:
            if dim in response_dict:
                if isinstance(response_dict[dim], dict):
                    scores[dim] = response_dict[dim].get("s", 0)
                else:
                    scores[dim] = int(response_dict[dim])
            else:
                scores[dim] = 0

        return scores

    @staticmethod
    def _compute_overall(scores: Dict[str, int]) -> float:
        """Compute normalized overall score (0-1)."""
        if not scores:
            return 0.0
        return sum(scores.values()) / (len(scores) * 5)
