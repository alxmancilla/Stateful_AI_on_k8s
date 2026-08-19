"""Hybrid search pipeline and Voyage AI reranker stage.

``build_rankfusion_pipeline`` assembles a single MongoDB 8 aggregation that
runs ``$vectorSearch`` and ``$search`` sub-pipelines and fuses them with the
``$rankFusion`` operator. ``VoyageReranker`` re-scores the fused candidates
with the Voyage AI Rerank API and logs the before/after ordering so the demo
can show the visible difference.
"""

from __future__ import annotations

import logging
import os
from functools import cached_property
from typing import Any, Dict, List

import voyageai

from embeddings import DEFAULT_VOYAGE_API_BASE, build_voyage_client

logger = logging.getLogger("agent.search")


def build_rankfusion_pipeline(
    query_vector: List[float],
    query_text: str,
    session_id: str,
    vector_index: str,
    text_index: str,
    top_k: int = 10,
    num_candidates: int = 100,
) -> List[Dict[str, Any]]:
    """Build a hybrid ``$rankFusion`` aggregation pipeline.

    The vector and full-text sub-pipelines are given equal weight. Session
    scoping is done inside each search stage (``filter`` for ``$vectorSearch``
    and a ``filter`` clause in the ``compound`` operator for ``$search``)
    because ``$rankFusion`` sub-pipelines may not contain a ``$limit`` stage.
    """
    vector_pipeline = [
        {
            "$vectorSearch": {
                "index": vector_index,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": num_candidates,
                "limit": top_k,
                "filter": {"session_id": {"$eq": session_id}},
            }
        }
    ]
    text_pipeline = [
        {
            "$search": {
                "index": text_index,
                "compound": {
                    "must": [{"text": {"query": query_text, "path": "content"}}],
                    "filter": [
                        {"equals": {"path": "session_id", "value": session_id}}
                    ],
                },
            }
        }
    ]
    return [
        {
            "$rankFusion": {
                "input": {
                    "pipelines": {
                        "vector": vector_pipeline,
                        "text": text_pipeline,
                    }
                },
                "combination": {"weights": {"vector": 1, "text": 1}},
                "scoreDetails": True,
            }
        },
        {"$limit": top_k},
        {
            "$project": {
                "_id": 1,
                "content": 1,
                "role": 1,
                "turn": 1,
                "timestamp": 1,
                "score": {"$meta": "score"},
            }
        },
    ]


class VoyageReranker:
    """Cross-encoder reranker backed by the Voyage AI Rerank API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ["VOYAGE_API_KEY"]
        self.model = model or os.environ.get("VOYAGE_RERANK_MODEL", "rerank-2.5")
        self.base_url = base_url or os.environ.get(
            "VOYAGE_API_BASE", DEFAULT_VOYAGE_API_BASE
        )

    @cached_property
    def _client(self) -> voyageai.Client:
        return build_voyage_client(self.api_key, self.base_url)

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """Re-score ``candidates`` and return the ``top_n`` most relevant.

        ``candidates`` is a list of memory documents; the ``content`` field of
        each is sent to the rerank endpoint in a single call.
        """
        if not candidates:
            return []

        documents = [c.get("content", "") for c in candidates]
        logger.info("Rerank BEFORE (hybrid order):")
        for i, doc in enumerate(documents):
            logger.info("  %2d. %s", i + 1, doc[:80])

        response = self._client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_k=top_n,
        )

        reranked: List[Dict[str, Any]] = []
        logger.info("Rerank AFTER (reranked order):")
        for rank, item in enumerate(response.results):
            candidate = dict(candidates[item.index])
            candidate["rerank_score"] = item.relevance_score
            reranked.append(candidate)
            logger.info(
                "  %2d. (%.4f) %s",
                rank + 1,
                item.relevance_score,
                candidate.get("content", "")[:80],
            )
        return reranked
