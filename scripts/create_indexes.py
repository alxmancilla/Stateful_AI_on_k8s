"""Create the MongoDB Search and Vector Search indexes for the agent.

Creates two search indexes on ``agent_memory.messages`` and blocks until both
report a ``READY`` / queryable status:

* ``vector_index`` - vector search on ``embedding`` (cosine, 1024 dims).
* ``text_index``   - BM25 full-text search on ``content`` (standard analyzer).

Run inside the cluster (it needs network access to ``mongot``), e.g.:

    kubectl exec deploy/stateful-agent -n mongodb -- python create_indexes.py
"""

from __future__ import annotations

import os
import sys
import time

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

DB_NAME = os.environ.get("DB_NAME", "agent_memory")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "messages")
VECTOR_INDEX = os.environ.get("VECTOR_INDEX", "vector_index")
TEXT_INDEX = os.environ.get("TEXT_INDEX", "text_index")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))
CA_FILE = os.environ.get("MONGODB_TLS_CA_FILE", "")


def get_collection():
    uri = os.environ["MONGODB_URI"]
    kwargs = {"appname": "create-indexes"}
    if CA_FILE and os.path.exists(CA_FILE):
        kwargs["tls"] = True
        kwargs["tlsCAFile"] = CA_FILE
    client = MongoClient(uri, **kwargs)
    db = client[DB_NAME]
    if COLLECTION_NAME not in db.list_collection_names():
        db.create_collection(COLLECTION_NAME)
    return db[COLLECTION_NAME]


def existing_index_names(collection) -> set:
    return {idx["name"] for idx in collection.list_search_indexes()}


def create_indexes(collection) -> None:
    names = existing_index_names(collection)

    if VECTOR_INDEX not in names:
        collection.create_search_index(
            SearchIndexModel(
                name=VECTOR_INDEX,
                type="vectorSearch",
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": EMBEDDING_DIM,
                            "similarity": "cosine",
                        },
                        {"type": "filter", "path": "session_id"},
                    ]
                },
            )
        )
        print(f"Created vector index '{VECTOR_INDEX}'.")
    else:
        print(f"Vector index '{VECTOR_INDEX}' already exists.")

    if TEXT_INDEX not in names:
        collection.create_search_index(
            SearchIndexModel(
                name=TEXT_INDEX,
                type="search",
                definition={
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            "content": {"type": "string", "analyzer": "lucene.standard"},
                            "session_id": {"type": "token"},
                        },
                    }
                },
            )
        )
        print(f"Created text index '{TEXT_INDEX}'.")
    else:
        print(f"Text index '{TEXT_INDEX}' already exists.")


def wait_until_ready(collection, timeout: int = 300) -> None:
    targets = {VECTOR_INDEX, TEXT_INDEX}
    deadline = time.time() + timeout
    while time.time() < deadline:
        statuses = {
            idx["name"]: idx.get("status", "UNKNOWN")
            for idx in collection.list_search_indexes()
            if idx["name"] in targets
        }
        print(f"index status: {statuses}")
        if targets.issubset(statuses) and all(
            s == "READY" for s in statuses.values()
        ):
            print("All search indexes are READY.")
            return
        time.sleep(5)
    raise TimeoutError(f"Search indexes not READY within {timeout}s")


def main() -> int:
    collection = get_collection()
    create_indexes(collection)
    wait_until_ready(collection)
    return 0


if __name__ == "__main__":
    sys.exit(main())
