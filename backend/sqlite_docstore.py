# sqlite_docstore.py

import sqlite3
import json
import logging
import time
from typing import List, Dict, Optional, Any
from pathlib import Path

from llama_index.core.schema import Document

log = logging.getLogger(__name__)


class SQLiteDocStore:
    def __init__(self, db_path: str, auto_connect: bool = True):
        self.db_path = db_path
        self._conn = None
        if auto_connect:
            self.connect()

    def __enter__(self):
        if not self._conn:
            self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the database connection explicitly."""
        if self._conn:
            log.debug("Closing database connection.")
            self._conn.close()
            self._conn = None
        else:
            log.warning("Database connection already closed.")

    def _init_table(self) -> None:
        """Ensure the table exists only once."""
        try:
            with self._conn:
                self._conn.execute("""
                    CREATE TABLE IF NOT EXISTS docstore (
                        doc_id TEXT PRIMARY KEY,
                        doc_json TEXT NOT NULL
                    )
                """)
        except Exception as e:
            log.error(f"Error initializing table: {e}")

    def _get_doc_id(self, doc: Any) -> Optional[str]:
        for attr in ("doc_id", "ref_doc_id", "id_"):
            doc_id = getattr(doc, attr, None)
            if doc_id:
                return doc_id
        log.warning(f"Could not extract doc_id from doc of type: {type(doc)}")
        return None

    def connect(self):
        """Connect to the SQLite database."""
        if self._conn:
            log.debug("Database connection already exists.")
            return
        # Retry logic for locked database
        retries = 3
        last_exception = None
        for attempt in range(retries):
            try:
                # Ensure the directory exists before attempting to connect
                if not Path(self.db_path).parent.exists():
                    log.error(f"The directory {Path(self.db_path).parent} does not exist.")
                    raise FileNotFoundError(f"The directory {Path(self.db_path).parent} does not exist.")
                
                # Check if the database file exists before attempting to connect
                if not Path(self.db_path).exists():
                    log.error(f"Database file {self.db_path} does not exist.")
                    raise FileNotFoundError(f"Database file {self.db_path} does not exist.")

                self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
                self._conn.row_factory = sqlite3.Row
                self._init_table()
                break
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    delay = 2 ** attempt
                    time.sleep(delay)
                    log.debug(f"Retrying... attempt {attempt + 1} took {delay} seconds.")
                else:
                    last_exception = e
                    break
        else:
            if last_exception:
                log.error(f"Failed to connect after {retries} retries: {last_exception}")
                raise last_exception

    def is_connected(self) -> bool:
        """Check if the database connection is open."""
        if not hasattr(self, '_conn') or self._conn is None:
            log.error("Database connection is closed unexpectedly.")
            raise sqlite3.ProgrammingError("Database connection is closed.")
        log.debug("Database connection is open.")
        return True

    def exists(self, doc_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM docstore WHERE doc_id = ? LIMIT 1", (doc_id,)
        ).fetchone()
        return row is not None

    def execute_query(self, query, params=()):
        """Execute raw SQL query with parameters."""
        return self._conn.execute(query, params)

    def add(self, doc_id: str, doc_dict: Dict[str, Any]) -> None:
        existing = self.get(doc_id)

        action = "Overwriting" if existing else "Adding new"
        log.debug(f"{action} doc for {doc_id}")

        if existing == doc_dict:
            log.debug(f"Doc for {doc_id} unchanged, skipping.")
            return

        doc_json = json.dumps(doc_dict)

        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO docstore (doc_id, doc_json) VALUES (?, ?)",
                (doc_id, doc_json),
            )

    def add_documents(self, docs: List[Document], **kwargs) -> None:
        with self._conn:
            for doc in docs:
                doc_id = self._get_doc_id(doc)
                if doc_id is None:
                    raise ValueError(f"Unsupported document type: {type(doc)}")
                doc_dict = doc.dict() if hasattr(doc, "dict") else doc.to_dict()
                try:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO docstore (doc_id, doc_json) VALUES (?, ?)",
                        (doc_id, json.dumps(doc_dict)),
                    )
                except sqlite3.DatabaseError as e:
                    log.error(f"Error adding document {doc_id}: {e}")
                    continue  # Skip over this document

    def set_document_hash(self, doc_id: str, doc_hash: str):
        log.debug(f"Setting document hash for {doc_id}")
        with self._conn:
            self._conn.execute(
                """
                UPDATE docstore
                SET doc_json = json_set(doc_json, '$.hash', ?)
                WHERE doc_id = ? 
                """,
                (doc_hash, doc_id),
            )

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT doc_json FROM docstore WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        try:
            return json.loads(row[0]) if row else None
        except json.JSONDecodeError as e:
            log.error(f"Failed to load doc {doc_id}: {e}")
            return None

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        rows = self._conn.execute("SELECT doc_id, doc_json FROM docstore").fetchall()
        return {doc_id: json.loads(doc_json) for doc_id, doc_json in rows}

    def get_document(self, doc_id: str) -> Optional[Document]:
        doc_dict = self.get(doc_id)
        if not doc_dict:
            log.warning(f"Document with doc_id {doc_id} not found.")
            return None
        return Document.from_dict(doc_dict)

    def get_all_docs(self, batch_size=100) -> Dict[str, Document]:
        docs = {}
        offset = 0
        while True:
            rows = self._conn.execute("SELECT doc_id FROM docstore LIMIT ? OFFSET ?", (batch_size, offset)).fetchall()
            if not rows:
                break
            for row in rows:
                doc_dict = self.get(row[0])
                if doc_dict:
                    docs[row[0]] = Document.from_dict(doc_dict)
            offset += batch_size
        return docs

    def get_all_doc_ids(self) -> List[str]:
        rows = self._conn.execute("SELECT doc_id FROM docstore").fetchall()
        return [row[0] for row in rows]

    def get_doc_ids_by_filename(self, filename: str) -> List[str]:
        rows = self._conn.execute(
            "SELECT doc_id FROM docstore WHERE json_extract(doc_json, '$.metadata.filename') = ?",
            (filename,)
        ).fetchall()
        return [row[0] for row in rows]

    def delete(self, doc_id: str) -> None:
        log.debug(f"Deleting doc {doc_id} from docstore")
        with self._conn:
            self._conn.execute("DELETE FROM docstore WHERE doc_id = ?", (doc_id,))

    def delete_all(self) -> None:
        log.debug("Deleting all docs from docstore")
        with self._conn:
            self._conn.execute("DELETE FROM docstore")

    def check_index_consistency(self) -> bool:
        """Check index consistency by comparing the count of stored docs and indexed docs."""
        try:
            # Get the documents and their count in a single step
            stored_docs = self.get_all()  # Fetch all stored docs in one go
            stored_docs_count = len(stored_docs)

            indexed_docs = self.get_all_docs()  # Get indexed documents in one go
            indexed_docs_count = len(indexed_docs)

            if stored_docs_count != indexed_docs_count:
                log.warning(f"Index inconsistency detected: {stored_docs_count} stored docs, {indexed_docs_count} indexed docs")
                return False

            # Compare each document's consistency
            for doc_id in stored_docs:
                stored_doc = stored_docs.get(doc_id)
                indexed_doc = indexed_docs.get(doc_id)

                if stored_doc and indexed_doc:
                    # Check for hash consistency if available
                    stored_hash = stored_doc.get('hash')
                    indexed_hash = indexed_doc.metadata.get('hash')

                    if stored_hash != indexed_hash:
                        log.warning(f"Hash mismatch for doc_id {doc_id}: stored={stored_hash}, indexed={indexed_hash}")
                        return False
                else:
                    log.warning(f"Document {doc_id} is missing either from the stored or indexed set")
                    return False

            log.info("Index consistency check passed.")
            return True
        except Exception as e:
            log.error(f"Error checking index consistency: {e}")
            return False

    def cleanup_temp_files(self) -> None:
        """Clean up SQLite temporary files like WAL and journal files."""
        temp_files = [Path(self.db_path).with_suffix(".db-journal"), Path(self.db_path).with_suffix(".db-wal")]
        for temp_file in temp_files:
            if temp_file.exists():
                log.debug(f"Deleting temporary file: {temp_file}")
                temp_file.unlink()
            else:
                log.debug(f"Temporary file not found, skipping deletion: {temp_file}")

    def close(self):
        """Close the SQLite connection."""
        self.__exit__(None, None, None)

    def persist(self, *args, **kwargs):
        """No-op persist to satisfy LlamaIndex API."""
        pass
