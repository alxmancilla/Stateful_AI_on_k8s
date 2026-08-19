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

from pymongo import ASCENDING, DESCENDING, ReturnDocument, MongoClient
from pymongo.collection import Collection
from pymongo.errors import OperationFailure

from embeddings import VoyageEmbeddings
from search import VoyageReranker, build_rankfusion_pipeline

# Upper bound on how many recent turns a single read may request, so a large
# client-supplied ``n`` (e.g. via the unauthenticated ``/memory`` route) cannot
# trigger an unbounded scan.
MAX_RECENT = 100


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
        counters: Optional[Collection] = None,
        ttl_seconds: int = 0,
        max_turns_per_session: int = 0,
    ) -> None:
        self.collection = collection
        self.embeddings = embeddings
        self.reranker = reranker
        self.vector_index = vector_index
        self.text_index = text_index
        # Memory-lifecycle controls. ``ttl_seconds`` <= 0 disables the TTL index
        # (documents never auto-expire); ``max_turns_per_session`` <= 0 disables
        # the per-session retention trim. Both default off so the demo keeps all
        # history unless explicitly configured.
        self.ttl_seconds = ttl_seconds
        self.max_turns_per_session = max_turns_per_session
        # Per-session turn sequences live in a sibling collection so the counter
        # can be advanced atomically, independent of the message documents.
        self.counters = (
            counters if counters is not None else collection.database["counters"]
        )

    def _next_turn(self, session_id: str) -> int:
        """Atomically reserve the next turn number for ``session_id``.

        Uses ``findOneAndUpdate`` with ``$inc`` (upserting on first use) so
        concurrent writers to the same session each receive a distinct,
        monotonically increasing turn value with no read-then-write race.

        The turn is reserved before the message is embedded and inserted, so a
        failure later in ``save_message`` burns that turn: the sequence stays
        strictly increasing and never duplicates, but may contain gaps. Reads
        order by ``turn``, so gaps are harmless.
        """
        doc = self.counters.find_one_and_update(
            {"_id": session_id},
            {"$inc": {"seq": 1}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return doc["seq"] - 1

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        embedding: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Embed ``content`` and persist a memory document.

        A precomputed ``embedding`` may be supplied to avoid re-embedding text
        that was already embedded for retrieval in the same turn.
        """
        now = datetime.now(timezone.utc)
        document = {
            "_id": str(uuid.uuid4()),
            "session_id": session_id,
            "role": role,
            "content": content,
            "embedding": embedding
            if embedding is not None
            else self.embeddings.embed_query(content),
            # ``timestamp`` is a human-readable ISO string kept for display and
            # back-compat; ``created_at`` is a BSON Date so a TTL index can
            # expire the document (TTL only acts on Date-typed fields).
            "timestamp": now.isoformat(),
            "created_at": now,
            "turn": self._next_turn(session_id),
        }
        self.collection.insert_one(document)
        if self.max_turns_per_session > 0:
            self._trim_session(session_id)
        return document

    def _trim_session(self, session_id: str) -> None:
        """Delete the oldest turns beyond ``max_turns_per_session`` for a session.

        Best-effort retention: keeps the newest ``max_turns_per_session``
        documents (by ``turn``) and removes older ones. Turn numbers keep
        advancing regardless, so reads stay correctly ordered.
        """
        keep = self.max_turns_per_session
        cursor = (
            self.collection.find({"session_id": session_id}, {"_id": 1})
            .sort([("turn", DESCENDING)])
            .skip(keep)
        )
        stale = [d["_id"] for d in cursor]
        if stale:
            self.collection.delete_many({"_id": {"$in": stale}})

    def get_recent(self, session_id: str, n: int = 5) -> List[Dict[str, Any]]:
        """Return the last ``n`` turns in chronological order.

        ``n`` is clamped to ``[1, MAX_RECENT]`` so a large or non-positive
        client-supplied value cannot request an unbounded read.
        """
        n = max(1, min(n, MAX_RECENT))
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
        self,
        session_id: str,
        query: str,
        top_k: int = 10,
        query_vector: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Combined ``$vectorSearch`` + ``$search`` via ``$rankFusion``.

        A precomputed ``query_vector`` may be supplied to avoid re-embedding a
        query that was already embedded in the same turn.
        """
        if query_vector is None:
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
        """Create the supporting b-tree indexes used for recent-history reads.

        The ``(session_id, turn)`` index is unique so that even if a turn value
        were ever reused, the duplicate insert is rejected rather than accepted
        silently. If a legacy collection already holds duplicate pairs (written
        before the atomic counter existed), the unique build fails; fall back to
        a non-unique index so startup is not blocked.

        This b-tree index is created here (at agent startup), separately from
        the search/vector indexes built by ``scripts/create_indexes.py``.
        """
        keys = [("session_id", ASCENDING), ("turn", DESCENDING)]
        try:
            self.collection.create_index(keys, unique=True)
        except OperationFailure:
            self.collection.create_index(keys)
        self._ensure_ttl_index()

    def _ensure_ttl_index(self) -> None:
        """Reconcile the TTL index on ``created_at`` with ``ttl_seconds``.

        ``ttl_seconds`` > 0 creates (or re-expires) the index; ``<= 0`` drops
        any existing one so disabling the TTL via config actually stops expiry.
        Because MongoDB rejects an index rebuild that only changes
        ``expireAfterSeconds``, an existing TTL index whose expiry differs is
        dropped and recreated so the configured value always wins.
        """
        name = "created_at_ttl"
        if self.ttl_seconds <= 0:
            # Disabled: remove a previously-created TTL index if present.
            try:
                self.collection.drop_index(name)
            except OperationFailure:
                pass
            return
        try:
            self.collection.create_index(
                [("created_at", ASCENDING)],
                name=name,
                expireAfterSeconds=self.ttl_seconds,
            )
        except OperationFailure:
            # An index with the same name but a different expiry already exists.
            self.collection.drop_index(name)
            self.collection.create_index(
                [("created_at", ASCENDING)],
                name=name,
                expireAfterSeconds=self.ttl_seconds,
            )
