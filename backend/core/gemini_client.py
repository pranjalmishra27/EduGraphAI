"""
backend/core/gemini_client.py
Thin wrapper around google-genai SDK with retry, multimodal helpers.
"""
from __future__ import annotations

from typing import Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger(__name__)
_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        settings = get_settings()
        _client = genai.Client(api_key=settings.gemini_api_key)
        logger.info("Gemini client initialised", model=settings.gemini_model)
    return _client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10),
       retry=retry_if_exception_type(Exception), reraise=True)
async def generate_text(prompt: str, system_instruction: str = "") -> str:
    from google import genai
    from google.genai import types
    settings = get_settings()
    client = _get_client()
    full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
    response = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=full_prompt,
    )
    return response.text


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10),
       retry=retry_if_exception_type(Exception), reraise=True)
async def generate_with_image(prompt: str, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    from google import genai
    from google.genai import types
    settings = get_settings()
    client = _get_client()
    response = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ],
    )
    return response.text


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10),
       retry=retry_if_exception_type(Exception), reraise=True)
async def generate_with_document(prompt: str, file_bytes: bytes, mime_type: str = "application/pdf") -> str:
    from google import genai
    from google.genai import types
    settings = get_settings()
    client = _get_client()
    response = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
            prompt,
        ],
    )
    return response.text


async def embed_text(text: str) -> list[float]:
    from google import genai
    client = _get_client()
    result = await client.aio.models.embed_content(
        model="models/gemini-embedding-2",
        contents=text,
    )
    return result.embeddings[0].values


async def embed_texts(texts: list[str]) -> list[list[float]]:
    import asyncio
    tasks = [embed_text(t) for t in texts]
    return await asyncio.gather(*tasks)
