"""Type 3 Judge: Virtual app trajectory evaluation."""

from pathlib import Path
from typing import Optional, Dict

from .base import BaseJudge, EvaluationResult
from .prompts import get_eval_prompt


class Type3Judge(BaseJudge):
    """Judge for Type 3 (virtual app trajectory generation)."""

    @property
    def data_type(self) -> str:
        return "type3"

    def evaluate_sample(self, sample_path: Path) -> Optional[EvaluationResult]:
        """
        Evaluate a Type 3 sample (virtual app trajectory).

        Expected structure:
        sample_path/
            ├── initial_frame.png (or equivalent)
            ├── frame1.png
            ├── frame2.png
            ├── frame3.png
            ├── frame4.png
            └── frame5.png
        """
        folder_name = sample_path.name
        lang_device = sample_path.parent.name

        # Load frames
        frames = {}
        for i in range(6):
            if i == 0:
                frame_path = self._find_image(sample_path, suffix="initial")
                if not frame_path:
                    frame_path = self._find_image(sample_path, suffix="frame0")
            else:
                frame_path = self._find_image(sample_path, suffix=f"frame{i}")

            if frame_path:
                frame = self._load_image(frame_path)
                if frame:
                    frames[f"frame{i}"] = frame
            else:
                print(f"[skip] Missing frame{i} in {sample_path}")
                return None

        # Load dataset metadata
        if not self.dataset_root:
            print(f"[skip] dataset_root required for evaluation")
            return None

        meta_path = self.dataset_root / "03_trajectory_text_fictionalapp" / lang_device / folder_name / "meta_data.json"
        metadata = self._load_metadata(meta_path)

        if not metadata:
            print(f"[skip] No metadata at {meta_path}")
            return None

        # Extract instruction/description
        instruction = metadata.get("instruction") or metadata.get("caption", "")
        if not instruction:
            print(f"[skip] No instruction in metadata")
            return None

        # Evaluate
        try:
            print(f"[eval] {lang_device}/{folder_name}...")

            sample_data = {
                "frames": frames,
                "instruction": instruction,
            }

            # Call evaluator
            prompt = get_eval_prompt("type3", lang_device)
            scores_dict = self.judge_provider.evaluate(
                sample_data,
                prompt=prompt + f"\n\nInstruction: {instruction}",
                max_retries=3,
            )

            # Parse scores
            scores = self._parse_scores(scores_dict)
            overall = self._compute_overall(scores)

            result = EvaluationResult(
                sample_name=f"{lang_device}/{folder_name}",
                data_type="type3",
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
        # Type 3 uses: goal, logic, consistency, ui, quality
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
