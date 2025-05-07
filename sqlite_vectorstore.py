"""
SQLiteVectorStore provides a simple local vector store with hybrid search
capabilities, using SQLite for storage.

Hybrid search combines:
- Vector similarity (cosine distance)
- Keyword matching on metadata values + vector_id

Parameters:
    alpha (float): weight of vector similarity in hybrid scoring
    beta (float): weight of keyword matches in hybrid scoring
"""

import sqlite3
import json
import logging
from typing import List, Dict, Optional, Any, Union
from scipy.spatial.distance import cosine

from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryResult, VectorStoreQueryMode
from llama_index.core.schema import BaseNode

log = logging.getLogger(__name__)


class SQLiteVectorStore:

    def __init__(self, db_path: str, alpha: float = 0.7, beta: float = 0.3):
        self.db_path = db_path
        self.alpha = alpha
        self.beta = beta
        self.stores_text = True
        self._conn = sqlite3.connect(self.db_path)
        self._init_table()

    def _init_table(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vectorstore (
                    vector_id TEXT PRIMARY KEY,
                    embedding_json TEXT NOT NULL,
                    metadata_json TEXT
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vectorstore_id ON vectorstore(vector_id)"
            )

    def query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        if query.mode == VectorStoreQueryMode.HYBRID:
            results = self.hybrid_search(
                query_embedding=query.query_embedding,
                query_keywords=query.filters.get("keywords") if query.filters else None,
                top_k=query.similarity_top_k or 5,
                alpha=self.alpha, beta=self.beta
            )
            return VectorStoreQueryResult(
                nodes=[r["vector_id"] for r in results],
                similarities=[r["hybrid_score"] for r in results],
            )
        else:
            raise ValueError(f"Unsupported query mode: {query.mode}")

    def add(self, nodes: List[BaseNode], embeddings: Optional[List[List[float]]] = None, **kwargs) -> List[str]:
        """Add or update nodes with embeddings."""
        if embeddings is None:
            embeddings = [getattr(node, 'embedding', None) for node in nodes]
            if any(e is None for e in embeddings):
                raise ValueError("Missing embeddings in nodes AND no embeddings provided to add()")

        ids = []
        for node, embedding in zip(nodes, embeddings):
            if not isinstance(embedding, list):
                raise TypeError(f"Expected embedding list, got {type(embedding)}")
            vector_id = node.node_id
            metadata = node.metadata or {}
            self._add_single(vector_id, embedding, metadata)
            ids.append(vector_id)

        return ids

    def _add_single(self, vector_id: str, embedding: List[float], metadata: Optional[Dict] = None) -> None:
        row = self._conn.execute(
            "SELECT 1 FROM vectorstore WHERE vector_id = ?", (vector_id,)
        ).fetchone()

        action = "Overwriting" if row else "Adding new"
        log.debug(f"{action} embedding for {vector_id}")

        embedding_json = json.dumps(embedding)
        metadata_json = json.dumps(metadata or {})

        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO vectorstore (vector_id, embedding_json, metadata_json)
                VALUES (?, ?, ?)
                """,
                (vector_id, embedding_json, metadata_json),
            )

    def get(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a node's embedding and metadata by ID."""
        row = self._conn.execute(
            "SELECT embedding_json, metadata_json FROM vectorstore WHERE vector_id = ?", (vector_id,)
        ).fetchone()

        if not row:
            return None

        try:
            return {"embedding": json.loads(row[0]), "metadata": json.loads(row[1])}
        except Exception as e:
            log.error(f"Failed to decode row for {vector_id}: {e}")
            return None

    def delete(self, vector_id: str) -> None:
        """Delete a node by ID."""
        log.debug(f"Deleting embedding for {vector_id}")
        with self._conn:
            self._conn.execute("DELETE FROM vectorstore WHERE vector_id = ?", (vector_id,))

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Return all stored embeddings and metadata."""
        rows = self._conn.execute(
            "SELECT vector_id, embedding_json, metadata_json FROM vectorstore"
        ).fetchall()

        result = {}
        for vector_id, embedding_json, metadata_json in rows:
            try:
                result[vector_id] = {
                    "embedding": json.loads(embedding_json),
                    "metadata": json.loads(metadata_json),
                }
            except Exception as e:
                log.error(f"Failed to decode row for {vector_id}: {e}")

        return result

    def hybrid_search(
        self,
        query_embedding: List[float],
        query_keywords: Optional[Union[str, List[str]]] = None,
        vector_id_filter: Optional[Union[str, List[str]]] = None,
        top_k: int = 5,
        alpha: float = 0.7,
        beta: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Hybrid search using vector similarity and keyword matching."""
        vector_id_filters = []
        if isinstance(vector_id_filter, str):
            vector_id_filters = [vector_id_filter]
        elif isinstance(vector_id_filter, list):
            vector_id_filters = vector_id_filter

        rows = self._conn.execute(
            "SELECT vector_id, embedding_json, metadata_json FROM vectorstore"
        ).fetchall()
        log.debug(f"Applying vector_id filter: {vector_id_filters}")
        
        keywords = []
        if isinstance(query_keywords, str):
            keywords = query_keywords.lower().split()
        elif isinstance(query_keywords, list):
            keywords = [kw.lower() for kw in query_keywords]
        
        log.debug(f"Running hybrid query with keywords={keywords} top_k={top_k}")

        results = []
        vector_id_filters = [vf.lower() for vf in vector_id_filters] if vector_id_filters else []
        for vector_id, embedding_json, metadata_json in rows:
            vector_id_lc = vector_id.lower()
            if vector_id_filters and not any(vector_id_lc.startswith(vf) for vf in vector_id_filters):
                continue
            try:
                embedding = json.loads(embedding_json)
                metadata = json.loads(metadata_json)
            except Exception as e:
                log.error(f"Failed to decode row for {vector_id}: {e}")
                continue

            # Create keyword corpus for text matching
            keyword_parts: List[str] = [
                f"{k} {str(v)}".lower() for k, v in metadata.items()
            ]
            keyword_parts.append(vector_id_lc)
            keyword_corpus: str = " ".join(keyword_parts)

            keyword_score = sum(kw in keyword_corpus for kw in keywords)

            vector_score = 1 - cosine(query_embedding, embedding)
            hybrid_score = alpha * vector_score + beta * keyword_score

            results.append(
                {
                    "vector_id": vector_id,
                    "embedding": embedding,
                    "metadata": metadata,
                    "vector_score": vector_score,
                    "keyword_score": keyword_score,
                    "hybrid_score": hybrid_score,
                }
            )

        top_results = sorted(results, key=lambda x: x["hybrid_score"], reverse=True)[:top_k]

        log.debug("Top hybrid search results:")
        for res in top_results:
            log.debug(
                f"{res['vector_id']}: vector_score={res['vector_score']:.4f}, "
                f"keyword_score={res['keyword_score']}, hybrid_score={res['hybrid_score']:.4f}"
            )

        return top_results

    def persist(self, persist_path: str = None, fs: Any = None) -> None:
        """Commit changes to SQLite."""
        if self._conn:
            self._conn.commit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self) -> None:
        """Close the SQLite connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            