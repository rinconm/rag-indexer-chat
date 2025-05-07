from __future__ import annotations

import sqlite3
import json
import logging
from typing import Dict, Optional, Any, List

from llama_index.core.storage.index_store.types import IndexStruct
from llama_index.core.storage.index_store.types import BaseIndexStore
from .sqlite_docstore import SQLiteDocStore
from llama_index.core import StorageContext


log = logging.getLogger(__name__)

def index_from_dict(data: dict):
    struct_type = data.get("type")

    if struct_type == "vector_store":
        from llama_index.core.indices.vector_store.base import VectorStoreIndex, IndexDict
        index_struct = IndexDict.from_dict(data)
        return VectorStoreIndex(index_struct=index_struct)

    if struct_type == "list":
        from llama_index.core.indices.list.base import SummaryIndex, IndexList
        index_struct = IndexList.from_dict(data)
        return SummaryIndex(index_struct=index_struct)

    if struct_type == "tree":
        from llama_index.core.indices.tree.base import TreeIndex, IndexTree
        index_struct = IndexTree.from_dict(data)
        return TreeIndex(index_struct=index_struct)

    if struct_type == "keyword_table":
        from llama_index.core.indices.keyword_table.base import KeywordTableIndex, KeywordTable
        index_struct = KeywordTable.from_dict(data)
        return KeywordTableIndex(index_struct=index_struct)

    raise ValueError(f"Unknown index struct type: {struct_type}")

class IndexDictPatched(dict):
    def __init__(self, data:dict):
        super().__init__(data)
        self.index_id = data.get("index_id")
        self.summary = data.get("summary")
        self.nodes_dict = data.get("nodes_dict", {})
        self.doc_id_dict = data.get("doc_id_dict", {})
        self.embeddings_dict = data.get("embeddings_dict", {})

    def get_type(self):
        return "vector_store"

    def to_dict(self):
        return {
            "index_id": self.index_id,
            "summary": self.summary,
            "nodes_dict": self.nodes_dict,
            "doc_id_dict": self.doc_id_dict,
            "embeddings_dict": self.embeddings_dict,
            "type": self.get_type(),
        }

class IndexListPatched(dict):
    def __init__(self, data:dict):
        self._data = data  # store raw dict

    def get_type(self):
        return "list"

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def to_dict(self):
        return self._data

class IndexTreePatched(dict):
    def __init__(self, data:dict):
        self._data = data  # store raw dict

    def get_type(self):
        return "tree"

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def to_dict(self):
        return self._data

class KeywordTablePatched(dict):
    def __init__(self, data:dict):
        self._data = data  # store raw dict

    def get_type(self):
        return "keyword_table"

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def to_dict(self):
        return self._data

def index_struct_from_dict(data: dict):
    struct_type = data.get("type")
    log.debug(f"index_struct_from_dict: raw data={data}")
    log.debug(f"index_struct_from_dict: loaded type={struct_type}")

    if struct_type == "vector_store":
        return IndexDictPatched(data)

    if struct_type == "list":
        return IndexListPatched(data)

    if struct_type == "tree":
        return IndexTreePatched(data)

    if struct_type == "keyword_table":
        return KeywordTablePatched(data)

    # fallback to core parsing
    try:
        return index_from_dict(data)
    except ValueError as e:
        log.error(f"Failed to parse index struct: {e}")
        raise

    raise ValueError(f"Unknown index struct type: {struct_type}")

class SQLiteIndexStore(BaseIndexStore):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_table()

    def _init_table(self):
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS indexstore (
                    index_id TEXT PRIMARY KEY,
                    index_json TEXT NOT NULL
                )
                """
            )

    def exists(self, index_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM indexstore WHERE index_id = ? LIMIT 1", (index_id,)
        ).fetchone()
        return row is not None

    def add_index_struct(self, index_struct: IndexStruct, **kwargs) -> None:
        index_id = index_struct.index_id
        index_dict = index_struct.to_dict()

        # Inject required type manually
        index_dict["type"] = index_struct.get_type()

        existing = self.get_index_struct(index_id)

        if existing and existing.to_dict() == index_dict:
            log.debug(f"Index struct for {index_id} unchanged, skipping.")
            return

        log.debug("Saving index_struct %s", index_id)
        index_json = json.dumps(index_dict)

        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO indexstore (index_id, index_json)
                VALUES (?, ?)
                """,
                (index_id, index_json),
            )

    def get_index_struct(self, index_id: str) -> Optional[IndexStruct]:
        """Retrieve IndexStruct by ID from SQLite."""
        row = self._conn.execute(
            "SELECT index_json FROM indexstore WHERE index_id = ?", (index_id,)
        ).fetchone()
        if row:
            try:
                return index_struct_from_dict(json.loads(row[0]))
            except (json.JSONDecodeError, ValueError) as e:
                log.error(f"Failed to parse index struct {index_id}: {e}")
        return None

    def index_structs(self) -> List[IndexStruct]:
        rows = self._conn.execute(
            "SELECT index_id, index_json FROM indexstore"
        ).fetchall()
        result = []
        for row in rows:
            index_json = row["index_json"]
            index_struct = index_struct_from_dict(json.loads(index_json))
            log.debug(f"returning type={type(index_struct)}, value={index_struct}")
            result.append(index_struct)
        return result

    def get_index_structs_dict(self) -> Dict[str, IndexStruct]:
        rows = self._conn.execute(
            "SELECT index_id, index_json FROM indexstore"
        ).fetchall()
        return {
            index_id: index_struct_from_dict(json.loads(index_json))  # NOT index_from_dict!
            for index_id, index_json in rows
        }

    def get_index(self, index_id: str) -> Optional[Any]:
        row = self._conn.execute(
            "SELECT index_json FROM indexstore WHERE index_id = ?", (index_id,)
        ).fetchone()
        return index_from_dict(json.loads(row[0])) if row else None

    # for app-level usage
    def get_all_indices(self) -> Dict[str, Any]:
        rows = self._conn.execute(
            "SELECT index_id, index_json FROM indexstore"
        ).fetchall()
        return {
            index_id: index_from_dict(json.loads(index_json))
            for index_id, index_json in rows
        }

    def delete_all(self) -> None:
        log.debug("Deleting all index structs from indexstore")
        with self._conn:
            self._conn.execute("DELETE FROM indexstore")

    def delete_index_struct(self, index_id: str):
        log.debug(f"Deleting index struct for {index_id}")
        with self._conn:
            self._conn.execute(
                "DELETE FROM indexstore WHERE index_id = ?", (index_id,)
            )

    def persist(self, persist_path: Optional[str] = None, fs: Optional[Any] = None):
        log.debug("Persisting indexstore to SQLite (commit only)")
        self._conn.commit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

if __name__ == "__main__":
    storage_context = StorageContext.from_defaults(
        index_store=SQLiteIndexStore("./index/sqlite.db"),
        docstore=SQLiteDocStore("./index/sqlite.db"),
    )
