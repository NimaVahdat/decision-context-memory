"""Configuration objects for this approach."""

from __future__ import annotations

from pydantic import BaseModel


class OpenAIConfig(BaseModel):
    """OpenAI settings used for embeddings and LLM calls."""

    api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
