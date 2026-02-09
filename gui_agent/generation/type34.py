"""Type 3 & 4 generators: Trajectory text generation (fictional and real apps)."""

from pathlib import Path
from typing import Optional, List, Tuple
import json

from .base import BaseGenerator, PromptBuilder


class Type3Type4PromptBuilder(PromptBuilder):
    """Prompt builder for Type 3/4 (trajectory text generation)."""

    def build(self, sample_data: dict) -> str:
        """
        Build prompt for trajectory text-based generation.

        Supports both first-frame (text-to-image) and subsequent frames (image-to-image).

        Args:
            sample_data: Dictionary containing:
                - step: Current step (1-5)
                - lang_device: Language/device identifier
                - app_name: Application name (for step 1)
                - final_goal: Final task goal (for step 1)
                - visual_description: Visual description for current step
                - action: User action for steps 2-5

        Returns:
            Structured prompt for trajectory generation
        """
        step = sample_data.get("step", 1)
        lang_device = sample_data.get("lang_device", "english_phone")
        is_chinese = lang_device.startswith("chinese_")

        # Determine device type
        device_cn_en = self._get_device_type(lang_device)
        device_cn, device_en = device_cn_en

        if step == 1:
            # First frame: text-to-image
            return self._build_first_frame_prompt(
                is_chinese=is_chinese,
                app_name=sample_data.get("app_name", "App"),
                final_goal=sample_data.get("final_goal", "Complete the task"),
                visual_description=sample_data.get("visual_description", ""),
                device_cn_en=device_cn_en
            )
        else:
            # Subsequent frames: image-to-image
            return self._build_next_frame_prompt(
                is_chinese=is_chinese,
                action=sample_data.get("action", ""),
                visual_description=sample_data.get("visual_description", ""),
                step_num=step,
                device_cn_en=device_cn_en
            )

    @staticmethod
    def _get_device_type(lang_device: str) -> Tuple[str, str]:
        """Map lang_device to (device_cn, device_en)."""
        if "phone" in lang_device:
            return ("手机", "mobile phone")
        elif "computer" in lang_device:
            return ("电脑", "computer")
        else:
            return ("手机", "mobile phone")

    @staticmethod
    def _build_first_frame_prompt(
        is_chinese: bool,
        app_name: str,
        final_goal: str,
        visual_description: str,
        device_cn_en: Tuple[str, str]
    ) -> str:
        """Build first-frame prompt (text-to-image generation)."""
        device_cn, device_en = device_cn_en

        if is_chinese:
            quality_checklist = (
                "生成前检查清单：\n"
                "☐ 所有文本清晰可读，字体大小适中\n"
                "☐ 配色方案与描述完全匹配\n"
                f"☐ 布局合理，符合{device_cn}屏幕规范\n"
                "☐ 包含状态栏、导航栏等标准UI组件\n"
                "☐ 符合平台原生设计模式\n"
                "☐ 没有幻觉元素，所有组件都在描述中提及"
            )

            return f"""你是一位专业UI/UX设计师，请为以下应用生成{device_cn}端UI界面截图。

<应用信息>
应用名称：{app_name}
最终目标：{final_goal}
</应用信息>

<视觉描述>
{visual_description}
</视觉描述>

<设备类型>
{device_cn}
</设备类型>

<风格要求>
现代原生{device_cn}应用设计
</风格要求>

<质量检查>
{quality_checklist}
</质量检查>

<强制约束>
1. 必须严格遵循视觉描述，不添加未提及的UI元素
2. 所有文本、图标必须清晰可读
3. 状态栏、导航栏等固定元素必须完整
4. 所有界面文本必须使用中文
5. 仅生成UI截图，不包含任何文字说明或标注
6. 不得出现用户手指、光标等交互指示
</强制约束>

直接生成最终的UI截图。"""

        else:
            quality_checklist = (
                "Pre-generation Checklist:\n"
                "☐ All text is clearly readable with appropriate font size\n"
                "☐ Color scheme matches the description exactly\n"
                f"☐ Layout is reasonable and follows {device_en} screen specifications\n"
                "☐ Contains standard UI components (status bar, navigation bar, etc.)\n"
                "☐ Follows platform-native design patterns\n"
                "☐ No hallucinated elements, all components are mentioned in the description"
            )

            return f"""You are a professional UI designer. Generate a {device_en} UI screenshot for this application.

<Application Info>
App Name: {app_name}
Final Goal: {final_goal}
</Application Info>

<Visual Description>
{visual_description}
</Visual Description>

<Device Type>
{device_en}
</Device Type>

<Style Requirements>
Modern native {device_en} app design
</Style Requirements>

<Quality Checklist>
{quality_checklist}
</Quality Checklist>

<Strict Constraints>
1. MUST follow the visual description exactly, no hallucinated elements
2. All text and icons MUST be clearly readable
3. Fixed UI elements (status bar, nav bar) MUST be complete
4. All interface text MUST be in English
5. Generate ONLY the UI screenshot, no text descriptions or annotations
6. MUST NOT show user finger, cursor, or interaction indicators
</Strict Constraints>

Generate the final UI screenshot directly."""

    @staticmethod
    def _build_next_frame_prompt(
        is_chinese: bool,
        action: str,
        visual_description: str,
        step_num: int,
        device_cn_en: Tuple[str, str]
    ) -> str:
        """Build subsequent-frame prompt (image-to-image generation)."""
        device_cn, device_en = device_cn_en

        if is_chinese:
            quality_checklist = (
                "生成前检查清单：\n"
                "☐ 所有文本清晰可读，字体大小适中\n"
                "☐ 配色方案与参考图保持一致\n"
                f"☐ 布局合理，符合{device_cn}屏幕规范\n"
                f"☐ 固定元素(状态栏、导航栏等)与参考图保持一致\n"
                "☐ 符合平台原生设计模式\n"
                "☐ 准确体现用户操作的预期结果\n"
                "☐ 仅修改操作影响的UI组件，其他保持不变\n"
                "☐ 没有幻觉元素，所有改动都在描述中提及"
            )

            return f"""基于参考{device_cn}UI截图，生成用户执行以下操作后的新界面状态：

用户操作：{action}

步骤{step_num}的视觉描述：
{visual_description}

<设备类型>
{device_cn}
</设备类型>

<质量检查>
{quality_checklist}
</质量检查>

<强制约束>
1. 必须严格遵循视觉描述，仅修改操作影响的UI组件
2. 所有文本、图标必须清晰可读
3. 状态栏、导航栏等固定元素必须与参考图保持一致
4. 所有界面文本必须使用中文
5. 仅生成UI截图，不包含任何文字说明或标注
6. 不得出现用户手指、光标等交互指示
</强制约束>

直接生成最终的UI截图。"""

        else:
            quality_checklist = (
                "Pre-generation Checklist:\n"
                "☐ All text is clearly readable with appropriate font size\n"
                "☐ Color scheme matches the reference image\n"
                f"☐ Layout is reasonable and follows {device_en} screen specifications\n"
                f"☐ Fixed UI elements (status bar, nav bar) match the reference image\n"
                "☐ Follows platform-native design patterns\n"
                "☐ Accurately reflects the expected result of the user action\n"
                "☐ Only modify UI components affected by the action, keep others unchanged\n"
                "☐ No hallucinated elements, all changes are mentioned in the description"
            )

            return f"""Based on the reference {device_en} UI screenshot, generate the next state after the user performs this action:

Action: {action}

Visual Description for Step {step_num}:
{visual_description}

<Device Type>
{device_en}
</Device Type>

<Quality Check>
{quality_checklist}
</Quality Check>

<Strict Constraints>
1. MUST follow the visual description exactly, only modify UI components affected by the action
2. All text and icons MUST be clearly readable
3. Fixed UI elements (status bar, navigation bar, etc.) MUST match the reference image
4. All interface text MUST be in English
5. Generate ONLY the UI screenshot, no text descriptions or annotations
6. MUST NOT show user finger, cursor, or interaction indicators
</Strict Constraints>

Generate the final UI screenshot directly."""


class Type3Generator(BaseGenerator):
    """Generator for Type 3 (fictional app trajectories)."""

    def __init__(self, provider, output_dir: Path, dataset_root: Optional[Path] = None):
        super().__init__(provider, Type3Type4PromptBuilder(), output_dir)
        self.dataset_root = dataset_root

    @property
    def data_type(self) -> str:
        return "type3"

    def process_sample(self, sample_path: Path) -> Optional[Path]:
        """Process a Type 3 trajectory JSON file."""
        raise NotImplementedError("Type 3/4 trajectory generation requires trajectory JSON format.")


class Type4Generator(BaseGenerator):
    """Generator for Type 4 (real app trajectories)."""

    def __init__(self, provider, output_dir: Path, dataset_root: Optional[Path] = None):
        super().__init__(provider, Type3Type4PromptBuilder(), output_dir)
        self.dataset_root = dataset_root

    @property
    def data_type(self) -> str:
        return "type4"

    def process_sample(self, sample_path: Path) -> Optional[Path]:
        """Process a Type 4 trajectory JSON file."""
        raise NotImplementedError("Type 3/4 trajectory generation requires trajectory JSON format.")
