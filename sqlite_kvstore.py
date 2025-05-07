# SQLiteKVStore

import sqlite3
import json
import logging
from typing import Any, Optional

log = logging.getLogger(__name__)


class SQLiteKVStore:
    def __init__(self, persist_path: str):
        self.conn = sqlite3.connect(persist_path)
        self.cur = self.conn.cursor()
        self.cur.execute(
            "CREATE TABLE IF NOT EXISTS kvstore (key TEXT PRIMARY KEY, value TEXT)"
        )
        self.conn.commit()

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __getitem__(self, key: str) -> Any:
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def __setitem__(self, key: str, value: Any):
        self.set(key, value)

    def __delitem__(self, key: str):
        self.delete(key)

    def get(self, key: str) -> Optional[Any]:
        self.cur.execute("SELECT value FROM kvstore WHERE key=?", (key,))
        row = self.cur.fetchone()
        try:
            return json.loads(row[0]) if row else None
        except json.JSONDecodeError as e:
            log.error(f"Failed to decode value for key {key}: {e}")
            return None

    def set(self, key: str, value: Any) -> None:
        existing = self.get(key)

        if existing == value:
            log.debug(f"KV for {key} unchanged, skipping set.")
            return

        if existing:
            log.debug(f"Overwriting existing KV for {key}")
        else:
            log.debug(f"Adding new KV for {key}")

        log.debug(f"Stored KV {key}: {json.dumps(value, indent=2)}")

        value_json = json.dumps(value)
        self.cur.execute(
            "INSERT OR REPLACE INTO kvstore (key, value) VALUES (?, ?)",
            (key, value_json),
        )
        self.conn.commit()

    def values(self) -> list[Any]:
        self.cur.execute("SELECT value FROM kvstore")
        results = []
        for row in self.cur.fetchall():
            try:
                results.append(json.loads(row[0]))
            except json.JSONDecodeError as e:
                log.error(f"Failed to decode value: {e}")
        return results

    def items(self) -> list[tuple[str, Any]]:
        self.cur.execute("SELECT key, value FROM kvstore")
        results = []
        for key, value_json in self.cur.fetchall():
            try:
                value = json.loads(value_json)
                results.append((key, value))
            except json.JSONDecodeError as e:
                log.error(f"Failed to decode value for key {key}: {e}")
        return results

    def delete(self, key: str) -> None:
        log.debug(f"Deleting KV for {key}")
        self.cur.execute("DELETE FROM kvstore WHERE key=?", (key,))
        self.conn.commit()

    def keys(self) -> list[str]:
        self.cur.execute("SELECT key FROM kvstore")
        return [row[0] for row in self.cur.fetchall()]

    def persist(self, persist_path: Optional[str] = None, fs: Optional[Any] = None):
        if persist_path or fs:
            log.warning(f"Ignoring unused persist_path={persist_path} fs={fs}")
        log.debug("Persisting kvstore to SQLite (commit only)")
        self.conn.commit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

