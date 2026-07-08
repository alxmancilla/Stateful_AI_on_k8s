"""Voyage AI embedding helper.

Wraps the ``voyageai`` client and exposes a small, LangChain-compatible
surface (``embed_query`` / ``embed_documents``) so the rest of the agent does
not need to know about Voyage-specific details.
"""

from __future__ import annotations

import os
from functools import cached_property
from typing import List

import voyageai


class VoyageEmbeddings:
    """Thin wrapper around the Voyage AI embeddings API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ["VOYAGE_API_KEY"]
        self.model = model or os.environ.get("VOYAGE_MODEL", "voyage-3")
        self.dimensions = dimensions or int(os.environ.get("EMBEDDING_DIM", "1024"))
        self.base_url = base_url or os.environ.get(
            "VOYAGE_API_BASE", "https://ai.mongodb.com/v1"
        )

    @cached_property
    def _client(self) -> voyageai.Client:
        # Route calls through the MongoDB Atlas endpoint by default. voyageai
        # 0.3.2 has no base_url client parameter, so the module-level global is
        # set; Atlas model API keys are rejected by the api.voyageai.com host.
        voyageai.api_base = self.base_url
        return voyageai.Client(api_key=self.api_key)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        result = self._client.embed(
            [text], model=self.model, input_type="query"
        )
        return result.embeddings[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of documents."""
        if not texts:
            return []
        result = self._client.embed(
            list(texts), model=self.model, input_type="document"
        )
        return result.embeddings
