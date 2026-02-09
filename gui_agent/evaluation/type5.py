"""Type 5 Judge: Grounding/spatial reasoning evaluation."""

from pathlib import Path
from typing import Optional, Dict

from .base import BaseJudge, EvaluationResult
from .prompts import get_eval_prompt


class Type5Judge(BaseJudge):
    """Judge for Type 5 (grounding/spatial reasoning - single frame generation)."""

    @property
    def data_type(self) -> str:
        return "type5"

    def evaluate_sample(self, sample_path: Path) -> Optional[EvaluationResult]:
        """
        Evaluate a Type 5 sample (grounding task).

        Expected structure:
        sample_path/
            ├── model_name.png (generated image after tap)
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

        # Load dataset metadata (initial image and grounding info)
        if not self.dataset_root:
            print(f"[skip] dataset_root required for evaluation")
            return None

        meta_path = self.dataset_root / "05_grounding_data" / lang_device / folder_name / "meta_data.json"
        metadata = self._load_metadata(meta_path)

        if not metadata:
            print(f"[skip] No metadata at {meta_path}")
            return None

        # Load initial image
        rel_image = metadata.get("image")
        grounding_info = metadata.get("grounding", {})

        if not rel_image:
            print(f"[skip] No image reference in metadata")
            return None

        init_image_path = self.dataset_root / "05_grounding_data" / str(rel_image)
        init_image = self._load_image(init_image_path)

        if not init_image:
            print(f"[skip] Could not load initial image: {init_image_path}")
            return None

        # Extract grounding explanation
        grounding_explanation = metadata.get("grounding_explanation", "")

        # Evaluate
        try:
            print(f"[eval] {lang_device}/{folder_name}...")

            sample_data = {
                "images": {
                    "initial": init_image,
                    "generated": gen_image,
                },
                "grounding": grounding_info,
                "grounding_explanation": grounding_explanation,
            }

            # Call evaluator
            prompt = get_eval_prompt("type5", lang_device)
            eval_prompt = prompt
            if grounding_explanation:
                eval_prompt += f"\n\nExpected Effect: {grounding_explanation}"

            scores_dict = self.judge_provider.evaluate(
                sample_data,
                prompt=eval_prompt,
                max_retries=3,
            )

            # Parse scores
            scores = self._parse_scores(scores_dict)
            overall = self._compute_overall(scores)

            result = EvaluationResult(
                sample_name=f"{lang_device}/{folder_name}",
                data_type="type5",
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
        # Type 5 uses: goal, logic, consistency, ui, quality
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
