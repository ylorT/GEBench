"""Evaluation provider implementations."""

import base64
import io
import json
import time
from typing import Optional, Dict
import requests

from .base import BaseJudgeProvider


class GPT4oProvider(BaseJudgeProvider):
    """GPT-4o evaluation provider."""

    DEFAULT_API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_key: str, api_endpoint: Optional[str] = None):
        super().__init__(api_key, api_endpoint)
        if not api_endpoint:
            self.api_endpoint = self.DEFAULT_API_ENDPOINT

    @property
    def name(self) -> str:
        return "gpt4o"

    def evaluate(self, sample_data: Dict, **kwargs) -> Dict[str, any]:
        """Evaluate using GPT-4o vision API."""
        self.validate_config()

        # Build request with images
        messages = self._build_request(sample_data, **kwargs)

        # Make API call
        max_retries = kwargs.get("max_retries", 3)
        timeout = kwargs.get("timeout", 300)

        for attempt in range(max_retries):
            try:
                response = self._call_api(messages, timeout)
                return self._parse_response(response)
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Request failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed after {max_retries} attempts: {e}")

    def _build_request(self, sample_data: Dict, **kwargs) -> list:
        """Build request messages for GPT-4o."""
        prompt = kwargs.get("prompt", "Evaluate this GUI screenshot.")
        images = sample_data.get("images", {})  # Dict of name -> PIL Image

        content = [{"type": "text", "text": prompt}]

        # Add images
        for img_name, img_obj in images.items():
            if img_obj:
                b64_data = self._encode_image(img_obj)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_data}",
                        "detail": "high"
                    }
                })

        return [{"role": "user", "content": content}]

    def _call_api(self, messages: list, timeout: int) -> dict:
        """Call GPT-4o API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gpt-4-vision-preview",  # or "gpt-4o"
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.7,
        }

        response = requests.post(
            self.api_endpoint,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    def _parse_response(self, response: dict) -> Dict[str, any]:
        """Parse GPT-4o response."""
        try:
            text = response["choices"][0]["message"]["content"]
            # Try to extract JSON
            if "{" in text and "}" in text:
                json_str = text[text.index("{"):text.rindex("}") + 1]
                return json.loads(json_str)
            # Fallback: return as-is
            return {"raw": text}
        except Exception as e:
            raise ValueError(f"Could not parse response: {e}")

    @staticmethod
    def _encode_image(image) -> str:
        """Encode PIL Image to base64."""
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        return base64.b64encode(img_bytes.getvalue()).decode("utf-8")


JUDGE_PROVIDERS = {
    "gpt4o": GPT4oProvider,
}


def get_judge_provider(name: str, api_key: str, api_endpoint: Optional[str] = None) -> BaseJudgeProvider:
    """
    Get a judge provider instance.

    Args:
        name: Provider name ('gpt4o')
        api_key: API key for the provider
        api_endpoint: Optional custom API endpoint

    Returns:
        Provider instance
    """
    if name not in JUDGE_PROVIDERS:
        raise ValueError(f"Unknown judge provider: {name}. Available: {list(JUDGE_PROVIDERS.keys())}")

    return JUDGE_PROVIDERS[name](api_key=api_key, api_endpoint=api_endpoint)
