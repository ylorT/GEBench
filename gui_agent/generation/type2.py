"""Type 2 generator: Multi-step UI trajectory generation (6 frames)."""

from pathlib import Path
from typing import Optional

from .base import BaseGenerator, PromptBuilder


class Type2PromptBuilder(PromptBuilder):
    """Prompt builder for Type 2 (multi-step trajectories)."""

    def build(self, sample_data: dict) -> str:
        """
        Build prompt for step-by-step UI trajectory generation.

        Args:
            sample_data: Dictionary containing:
                - step: Current step number (1-5)
                - goal: Overall task goal
                - lang_device: Language/device identifier

        Returns:
            Step-specific prompt
        """
        step = sample_data.get("step", 1)
        goal = sample_data.get("goal", "Complete the task")
        lang_device = sample_data.get("lang_device", "english_phone")
        is_chinese = lang_device.startswith("chinese_")

        if is_chinese:
            return (
                f"目标：{goal}\n\n"
                f"进度：第 {step}/5 步\n\n"
                "说明：根据任务目标，生成本步交互后的界面。逐步演进：\n"
                "• 第1步：开始任务（如打开应用、进入入口）\n"
                "• 第2-4步：执行交互（如输入、点击、浏览）\n"
                "• 第5步：接近完成（如查看结果、确认）\n\n"
                "要求：\n"
                "1. 保持参考图的设计风格与布局一致\n"
                "2. 变化合理自然，符合真实移动端/桌面端交互\n"
                "3. 元素清晰可读，文本和按钮需清楚\n"
                "4. 与上一步有连贯的界面变化\n"
            )
        else:
            return (
                "Context:\n"
                "A mobile GUI agent is completing a task through multi-step interactions.\n\n"
                f"Goal: {goal}\n\n"
                f"Progress: Step {step} / 5\n\n"
                "Instruction:\n"
                f"Based on the task goal, generate the interface after step {step}. The agent is progressively advancing:\n"
                "• Step 1: Start the task (e.g., open app, activate search)\n"
                "• Steps 2-4: Execute interactions (e.g., input, click, browse)\n"
                "• Step 5: Approach completion (e.g., view results, confirm answer)\n\n"
                "Requirements:\n"
                "1. Match the design style and layout from the reference image\n"
                "2. Natural progression: Interface changes should be reasonable and gradual\n"
                "3. Clear visibility: Ensure all text, buttons, and UI elements are clearly readable\n"
                "4. Logical coherence: Show noticeable but not abrupt progress compared to the previous step\n"
            )


class Type2Generator(BaseGenerator):
    """Generator for Type 2 data (6-frame multi-step trajectories)."""

    def __init__(self, provider, output_dir: Path, dataset_root: Optional[Path] = None):
        super().__init__(provider, Type2PromptBuilder(), output_dir)
        self.dataset_root = dataset_root

    @property
    def data_type(self) -> str:
        return "type2"

    def process_sample(self, sample_path: Path) -> Optional[Path]:
        """
        Process a Type 2 sample (6-frame trajectory).

        Expected structure:
        sample_path/
            ├── metadata.jsonl or meta_data.json (contains: question, image_size, lang_device)
            └── <initial frame image>

        Output:
        output_dir/{lang_device}/{folder_name}/frame1.png ... frame5.png
        """
        # Load metadata
        meta_path = sample_path / "metadata.jsonl"
        if not meta_path.exists():
            meta_path = sample_path / "meta_data.json"

        metadata = self._load_metadata(meta_path)
        if not metadata:
            print(f"[skip] No metadata in {sample_path}")
            return None

        # Extract required fields
        goal = metadata.get("question") or metadata.get("caption", "Complete the task")
        lang_device = metadata.get("lang_device", sample_path.parent.name)
        folder_name = sample_path.name

        # Find initial image
        image_path = self._find_image(sample_path)
        if not image_path:
            print(f"[skip] No image found in {sample_path}")
            return None

        # Generate 5 frames
        output_dir = self.output_dir / lang_device / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if already completed (all 5 frames exist)
        final_output = output_dir / "frame5.png"
        if self._should_skip(final_output):
            print(f"[skip] Already processed: {final_output.relative_to(self.output_dir.parent)}")
            return final_output

        # Load initial image
        reference_image = self._load_image(image_path)
        if not reference_image:
            print(f"[skip] Could not load image: {image_path}")
            return None

        # Generate frames 1-5
        prev_image = reference_image
        for step_num in range(1, 6):
            frame_path = output_dir / f"frame{step_num}.png"

            # Skip if frame already exists
            if frame_path.exists() and frame_path.stat().st_size > 1024:
                print(f"[skip] frame{step_num} exists")
                prev_image = frame_path
                continue

            # Build step-specific prompt
            sample_data = {
                "step": step_num,
                "goal": goal,
                "lang_device": lang_device,
            }
            prompt = self.prompt_builder.build(sample_data)

            try:
                print(f"[gen] {lang_device}/{folder_name} frame{step_num}...")
                generated_image = self.provider.generate(
                    prompt=prompt,
                    reference_image=prev_image,
                )

                self._save_output(frame_path, generated_image)
                print(f"[done] frame{step_num}")
                prev_image = generated_image

            except Exception as e:
                print(f"[error] Failed to generate frame{step_num}: {e}")
                return None

        return final_output

    @staticmethod
    def _find_image(folder: Path) -> Optional[Path]:
        """Find first image file in folder."""
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
            candidates = sorted(folder.glob(f"*{ext}"))
            if candidates:
                return candidates[0]
        return None
