"""OpenAI embedding calls shared across all approaches."""

import logging

from openai import AsyncOpenAI

from doc_naive.shared.usage import record_embed

logger = logging.getLogger(__name__)


class Embedder:
    """Thin wrapper around OpenAI embeddings API."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        response = await self._client.embeddings.create(
            input=[text],
            model=self._model,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            record_embed(getattr(usage, "total_tokens", 0))
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in one API call."""
        if not texts:
            return []
        response = await self._client.embeddings.create(
            input=texts,
            model=self._model,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            record_embed(getattr(usage, "total_tokens", 0))
        return [item.embedding for item in response.data]
