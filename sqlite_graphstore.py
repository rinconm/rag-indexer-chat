# SQLiteGraphStore

import sqlite3
import json
import logging
from typing import Dict, Optional, Any

log = logging.getLogger(__name__)


class SQLiteGraphStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path)
        self._init_table()

    def _init_table(self):
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graphstore (
                    key TEXT PRIMARY KEY,
                    graph_json TEXT NOT NULL
                )
                """
            )

    def add(self, key: str, graph_dict: Dict):
        existing = self.get(key)

        if existing == graph_dict:
            log.debug(f"Graph for {key} unchanged, skipping add.")
            return

        if existing:
            log.debug(f"Overwriting existing graph for {key}")
        else:
            log.debug(f"Adding new graph for {key}")

        graph_json = json.dumps(graph_dict, indent=2)

        log.debug(f"Stored graph {key}: {graph_json}")

        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO graphstore (key, graph_json)
                VALUES (?, ?)
                """,
                (key, graph_json),
            )

    def get(self, key: str) -> Optional[Dict]:
        row = self._conn.execute(
            "SELECT graph_json FROM graphstore WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def delete(self, key: str):
        log.debug(f"Deleting graph for {key}")
        with self._conn:
            self._conn.execute("DELETE FROM graphstore WHERE key = ?", (key,))

    def get_all(self) -> Dict[str, Dict]:
        rows = self._conn.execute("SELECT key, graph_json FROM graphstore").fetchall()
        return {key: json.loads(graph_json) for key, graph_json in rows}

    def persist(self, persist_path: Optional[str] = None, fs: Optional[Any] = None):
        if persist_path is not None or fs is not None:
            raise ValueError(f"SQLiteGraphStore.persist() does not support persist_path or fs args. Got: persist_path={persist_path}, fs={fs}")

        log.debug("Persisting graphstore to SQLite (commit only)")
        self._conn.commit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self._conn.close()
