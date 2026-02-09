"""API provider implementations for generation."""

import base64
import io
import time
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

from .base import BaseProvider


class GeminiProvider(BaseProvider):
    """Gemini-3-Pro-Image provider implementation."""

    # Official Gemini API endpoint
    DEFAULT_API_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    def __init__(self, api_key: str, api_endpoint: Optional[str] = None):
        super().__init__(api_key, api_endpoint)
        if not api_endpoint:
            self.api_endpoint = self.DEFAULT_API_ENDPOINT

    @property
    def name(self) -> str:
        return "gemini"

    def validate_config(self) -> bool:
        """Validate Gemini configuration."""
        if not self.api_key:
            raise ValueError("Gemini API key not provided")
        return True

    def generate(
        self,
        prompt: str,
        reference_image: Optional[Image.Image] = None,
        **kwargs
    ) -> Image.Image:
        """
        Generate image using Gemini API.

        Args:
            prompt: Text prompt
            reference_image: Optional reference image for vision input
            **kwargs: Additional parameters (model, temperature, etc.)

        Returns:
            Generated PIL Image
        """
        self.validate_config()

        # Prepare request
        messages = self._build_request(prompt, reference_image, **kwargs)

        # Make API call with retry logic
        max_retries = kwargs.get("max_retries", 3)
        timeout = kwargs.get("timeout", 300)

        for attempt in range(max_retries):
            try:
                response = self._call_api(messages, timeout)
                return self._extract_image(response)
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Request failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed to generate image after {max_retries} attempts: {e}")

    def _build_request(self, prompt: str, reference_image: Optional[Image.Image], **kwargs) -> dict:
        """Build request payload for Gemini API."""
        content = []

        # Add reference image if provided
        if reference_image:
            img_base64 = self._encode_image(reference_image)
            content.append({
                "type": "image",
                "image": {
                    "format": "png",
                    "data": img_base64,
                }
            })

        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })

        return {
            "contents": [{
                "parts": content
            }]
        }

    def _call_api(self, messages: dict, timeout: int) -> dict:
        """Make API call to Gemini."""
        url = f"{self.api_endpoint}?key={self.api_key}"

        response = requests.post(
            url,
            json=messages,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()

    def _extract_image(self, response: dict) -> Image.Image:
        """Extract image data from API response."""
        try:
            # Navigate through response structure
            candidates = response.get("candidates", [])
            if not candidates:
                raise ValueError("No candidates in response")

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])

            for part in parts:
                if "inlineData" in part:
                    data = part["inlineData"].get("data", "")
                    mime_type = part["inlineData"].get("mimeType", "image/png")

                    if data:
                        # Decode base64 and convert to PIL Image
                        image_bytes = base64.b64decode(data)
                        image = Image.open(io.BytesIO(image_bytes))
                        return image

            raise ValueError("No image data found in response")
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"Failed to extract image from response: {e}")

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        """Encode PIL Image to base64."""
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Save to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # Encode to base64
        return base64.b64encode(img_bytes.getvalue()).decode("utf-8")


PROVIDERS = {
    "gemini": GeminiProvider,
}


def get_provider(name: str, api_key: str, api_endpoint: Optional[str] = None) -> BaseProvider:
    """
    Get a provider instance.

    Args:
        name: Provider name ('gemini', 'gpt', 'seedream')
        api_key: API key for the provider
        api_endpoint: Optional custom API endpoint

    Returns:
        Provider instance
    """
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDERS.keys())}")

    return PROVIDERS[name](api_key=api_key, api_endpoint=api_endpoint)
