"""Type 5 generator: Grounding/spatial reasoning generation."""

from pathlib import Path
from typing import Optional

from .base import BaseGenerator, PromptBuilder


class Type5PromptBuilder(PromptBuilder):
    """Prompt builder for Type 5 (grounding/spatial reasoning)."""

    def build(self, sample_data: dict) -> str:
        """
        Build prompt for grounding/spatial task generation.

        Handles grounding metadata (point/box coordinates) normalized to [0,1000] range.

        Args:
            sample_data: Dictionary containing:
                - metadata: Full metadata dict with 'grounding', 'width', 'height'
                - lang_device: "chinese_phone", "english_computer", etc.

        Returns:
            Prompt with normalized coordinate information
        """
        lang_device = sample_data.get("lang_device", "english_phone")
        is_chinese = lang_device.startswith("chinese_")
        metadata = sample_data.get("metadata", {})

        # Extract grounding information
        width = int(metadata.get("width", 0) or 0)
        height = int(metadata.get("height", 0) or 0)
        grounding = metadata.get("grounding", {}) or {}
        gtype = str(grounding.get("type", "point")).lower()

        # Parse point or box coordinates
        px = py = None
        if gtype == "point":
            pt = grounding.get("point") or []
            if isinstance(pt, (list, tuple)) and len(pt) == 2:
                px, py = pt[0], pt[1]
        elif gtype in ("box", "rectangle"):
            box = grounding.get("box") or grounding.get("rectangle") or []
            if isinstance(box, (list, tuple)) and len(box) == 4:
                x1, y1, x2, y2 = box
                px = (float(x1) + float(x2)) / 2.0
                py = (float(y1) + float(y2)) / 2.0

        # Normalize to [0,1000] range
        if px is None or py is None:
            nx, ny = 500, 500
        else:
            if width <= 0 or height <= 0:
                nx = int(round(float(px)))
                ny = int(round(float(py)))
            else:
                nx = int(round(float(px) / float(width) * 1000.0))
                ny = int(round(float(py) / float(height) * 1000.0))
            nx = max(0, min(1000, nx))
            ny = max(0, min(1000, ny))

        point_json = f'{{\"point\": [{nx}, {ny}]}}'

        if is_chinese:
            return (
                "请基于提供的参考图生成下一帧的预测图片：\n"
                f"交互输入： 用户在屏幕上执行了点击操作，点击位置的归一化坐标为 {point_json}（坐标范围归一化至[0,1000]，原点左上角，x向右，y向下）。\n\n"
                "任务要求：\n"
                "1) 识别该坐标在原图中所对应的 UI 元素。\n"
                "2) 预测并生成点击该元素后，界面发生的即时视觉变化（下一帧）。\n"
                "3) 保持页面其他部分的视觉一致性，仅展示交互触发的动态效果（如弹出层、菜单或状态切换）。\n"
            )
        else:
            return (
                "Please generate the next-frame prediction based on the provided reference image.\n"
                f"Interaction input: The user performed a tap; the normalized relative coordinate is {point_json} (normalized to [0,1000], origin at top-left, x→right, y→down).\n\n"
                "Task requirements:\n"
                "1) Identify the UI element at this coordinate in the reference image.\n"
                "2) Predict and render the immediate visual change after tapping this element (the next frame).\n"
                "3) Preserve visual consistency elsewhere; show only interaction-triggered dynamics (e.g., popup, menu, or state toggle).\n"
            )


class Type5Generator(BaseGenerator):
    """Generator for Type 5 (grounding/spatial reasoning)."""

    def __init__(self, provider, output_dir: Path, dataset_root: Optional[Path] = None):
        super().__init__(provider, Type5PromptBuilder(), output_dir)
        self.dataset_root = dataset_root

    @property
    def data_type(self) -> str:
        return "type5"

    def process_sample(self, sample_path: Path) -> Optional[Path]:
        """
        Process a Type 5 sample (grounding task with coordinate-based interaction).

        Expected structure:
        sample_path/
            ├── meta_data.json (contains: grounding, width, height, lang_device)
            └── <image file>

        Output:
        output_dir/{lang_device}/{folder_name}/{provider}.png
        """
        # Load metadata
        meta_path = sample_path / "meta_data.json"
        metadata = self._load_metadata(meta_path)

        if not metadata:
            print(f"[skip] No metadata in {sample_path}")
            return None

        # Extract fields
        lang_device = metadata.get("lang_device", sample_path.parent.name)
        folder_name = sample_path.name

        # Find image
        image_path = self._find_image(sample_path)
        if not image_path:
            print(f"[skip] No image found in {sample_path}")
            return None

        # Output path
        output_path = self.output_dir / lang_device / folder_name / f"{self.provider.name}.png"

        if self._should_skip(output_path):
            print(f"[skip] Already processed: {output_path.relative_to(self.output_dir.parent)}")
            return output_path

        # Load image
        reference_image = self._load_image(image_path)
        if not reference_image:
            print(f"[skip] Could not load image: {image_path}")
            return None

        # Build and generate
        sample_data = {
            "metadata": metadata,
            "lang_device": lang_device,
        }
        prompt = self.prompt_builder.build(sample_data)

        try:
            print(f"[gen] {lang_device}/{folder_name}...")
            generated_image = self.provider.generate(
                prompt=prompt,
                reference_image=reference_image,
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
