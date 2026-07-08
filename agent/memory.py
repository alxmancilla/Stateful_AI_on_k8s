"""MongoDB-backed conversational memory.

Each turn is stored as a single document in ``agent_memory.messages`` with the
schema described in the README. This module owns all reads/writes and exposes
recent-history, semantic, and hybrid retrieval helpers.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection

from embeddings import VoyageEmbeddings
from search import VoyageReranker, build_rankfusion_pipeline


def get_client(uri: str, ca_file: Optional[str] = None) -> MongoClient:
    """Create a MongoClient, enabling TLS when a CA file is provided."""
    kwargs: Dict[str, Any] = {"appname": "stateful-agent"}
    if ca_file and os.path.exists(ca_file):
        kwargs["tls"] = True
        kwargs["tlsCAFile"] = ca_file
    return MongoClient(uri, **kwargs)


class MongoMemory:
    """Read/write layer for agent memory documents."""

    def __init__(
        self,
        collection: Collection,
        embeddings: VoyageEmbeddings,
        reranker: VoyageReranker,
        vector_index: str = "vector_index",
        text_index: str = "text_index",
    ) -> None:
        self.collection = collection
        self.embeddings = embeddings
        self.reranker = reranker
        self.vector_index = vector_index
        self.text_index = text_index

    def _next_turn(self, session_id: str) -> int:
        last = self.collection.find_one(
            {"session_id": session_id}, sort=[("turn", DESCENDING)]
        )
        return (last["turn"] + 1) if last else 0

    def save_message(self, session_id: str, role: str, content: str) -> Dict[str, Any]:
        """Embed ``content`` and persist a memory document."""
        document = {
            "_id": str(uuid.uuid4()),
            "session_id": session_id,
            "role": role,
            "content": content,
            "embedding": self.embeddings.embed_query(content),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "turn": self._next_turn(session_id),
        }
        self.collection.insert_one(document)
        return document

    def get_recent(self, session_id: str, n: int = 5) -> List[Dict[str, Any]]:
        """Return the last ``n`` turns in chronological order."""
        cursor = (
            self.collection.find(
                {"session_id": session_id}, {"embedding": 0}
            )
            .sort([("turn", DESCENDING)])
            .limit(n)
        )
        return list(cursor)[::-1]

    def semantic_search(
        self, session_id: str, query: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Pure ``$vectorSearch`` retrieval scoped to a session."""
        query_vector = self.embeddings.embed_query(query)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.vector_index,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": max(top_k * 10, 100),
                    "limit": top_k,
                    "filter": {"session_id": {"$eq": session_id}},
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "content": 1,
                    "role": 1,
                    "turn": 1,
                    "timestamp": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        return list(self.collection.aggregate(pipeline))

    def hybrid_search(
        self, session_id: str, query: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Combined ``$vectorSearch`` + ``$search`` via ``$rankFusion``."""
        query_vector = self.embeddings.embed_query(query)
        pipeline = build_rankfusion_pipeline(
            query_vector=query_vector,
            query_text=query,
            session_id=session_id,
            vector_index=self.vector_index,
            text_index=self.text_index,
            top_k=top_k,
        )
        return list(self.collection.aggregate(pipeline))

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """Re-score hybrid candidates with the Voyage AI reranker."""
        return self.reranker.rerank(query, candidates, top_n=top_n)

    def ensure_indexes(self) -> None:
        """Create the supporting b-tree indexes used for recent-history reads."""
        self.collection.create_index(
            [("session_id", ASCENDING), ("turn", DESCENDING)]
        )
