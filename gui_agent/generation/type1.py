"""Type 1 generator: Single-step UI transition generation."""

from pathlib import Path
from typing import Optional

from PIL import Image

from .base import BaseGenerator, PromptBuilder


class Type1PromptBuilder(PromptBuilder):
    """Prompt builder for Type 1 (single-step transitions)."""

    def build(self, sample_data: dict) -> str:
        """
        Build prompt for single-step UI generation.

        Args:
            sample_data: Dictionary containing:
                - lang_device: "chinese_phone", "english_computer", etc.
                - caption: User action description (REQUIRED for Type 1)

        Returns:
            Language-aware prompt
        """
        caption = sample_data.get("caption", "").strip()
        lang_device = sample_data.get("lang_device", "english_phone")
        is_chinese = lang_device.startswith("chinese_")

        if is_chinese:
            return (
                "根据第一张截图（参考界面）和以下说明，生成第二张截图，"
                "表现用户交互后的新界面状态：\n"
                f"{caption}\n\n"
                "要求：\n"
                "- 保持原有风格与布局一致，元素清晰可读\n"
                "- 变化合理自然，符合真实应用逻辑\n"
                "- 分辨率与参考图一致"
            )
        else:
            return (
                "Using the first screenshot as reference, generate a second screenshot"
                " showing the NEW UI state after user interaction:\n"
                f"{caption}\n\n"
                "Requirements:\n"
                "- Preserve style/layout, keep all elements readable\n"
                "- Changes should be natural and coherent\n"
                "- Match the reference resolution"
            )


class Type1Generator(BaseGenerator):
    """Generator for Type 1 data (single-step UI transitions)."""

    def __init__(self, provider, output_dir: Path, dataset_root: Optional[Path] = None):
        super().__init__(provider, Type1PromptBuilder(), output_dir)
        self.dataset_root = dataset_root

    @property
    def data_type(self) -> str:
        return "type1"

    def process_sample(self, sample_path: Path) -> Optional[Path]:
        """
        Process a Type 1 sample (single screenshot transition).

        Expected structure:
        sample_path/
            ├── meta_data.json (contains: image, caption, lang_device, etc.)
            └── <image file>

        Output:
        output_dir/{lang_device}/{folder_name}/gemini.png (or provider.png)
        """
        # Load metadata
        meta_path = sample_path / "meta_data.json"
        metadata = self._load_metadata(meta_path)

        if not metadata:
            print(f"[skip] No metadata in {sample_path}")
            return None

        # Extract required fields
        caption = metadata.get("caption", "").strip()
        lang_device = metadata.get("lang_device", sample_path.parent.name)
        folder_name = sample_path.name

        # Find image file
        image_path = self._find_image(sample_path)
        if not image_path:
            print(f"[skip] No image found in {sample_path}")
            return None

        # Determine output path
        output_path = self.output_dir / lang_device / folder_name / f"{self.provider.name}.png"

        # Check if already processed
        if self._should_skip(output_path):
            print(f"[skip] Already processed: {output_path.relative_to(self.output_dir.parent)}")
            return output_path

        # Load image
        image = self._load_image(image_path)
        if not image:
            print(f"[skip] Could not load image: {image_path}")
            return None

        # Build prompt
        sample_data = {
            "caption": caption,
            "lang_device": lang_device,
        }
        prompt = self.prompt_builder.build(sample_data)

        try:
            print(f"[gen] {lang_device}/{folder_name}...")
            generated_image = self.provider.generate(
                prompt=prompt,
                reference_image=image,
            )

            self._save_output(output_path, generated_image)
            print(f"[done] {output_path.relative_to(self.output_dir.parent)}")
            return output_path

        except Exception as e:
            print(f"[error] Failed to generate {lang_device}/{folder_name}: {e}")
            return None

    @staticmethod
    def _find_image(folder: Path) -> Optional[Path]:
        """Find first image file in folder."""
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
            candidates = sorted(folder.glob(f"*{ext}"))
            if candidates:
                return candidates[0]
        return None
