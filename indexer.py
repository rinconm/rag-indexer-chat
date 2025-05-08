# indexer.py
# Retrieve-Augmented Generator (RAG) Indexer

import sys
import time
import os
import json
import signal
import logging
import argparse
import shutil
import sqlite3
from textwrap import dedent
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import Counter
import re

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)

# Combine
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
try:
    from llama_index.core.indices.base import IndexNotFoundError
except ImportError:
    # fallback for older versions
    IndexNotFoundError = Exception

from backend.sqlite_kvstore import SQLiteKVStore
from backend.sqlite_docstore import SQLiteDocStore
from backend.sqlite_indexstore import SQLiteIndexStore
from backend.sqlite_vectorstore import SQLiteVectorStore
from backend.sqlite_graphstore import SQLiteGraphStore

# --- LlamaIndex Global Settings ---
from llama_index.core.settings import Settings

@dataclass(frozen=True)
class Config:
    index_dir   : Path = Path("./index")
    sqlite_path : Path = index_dir / "sqlite.db"
    cache_path  : Path = index_dir / ".index_cache.json"
    supported_extensions: frozenset = frozenset({
        ".py", ".sh", ".md", ".txt", ".yaml", ".yml", ".json",
        ".html", ".js", ".ts", ".css", ".java", ".cpp", ".c", ".go"
    })
    
def configure_settings(model: str, parser: SimpleNodeParser) -> None:
    """Configure LLM, embedding, and node parser settings."""
    Settings.store_text = True
    Settings.embed_model = HuggingFaceEmbedding(model_name=model)
    Settings.llm = Ollama(model="mistral")
    Settings.node_parser = parser

def cleanup_deleted(docstore: SQLiteDocStore, file_cache: Dict[str, float], updated_cache: Dict[str, float]) -> set:
    """Remove deleted files from docstore."""
    deleted_files = set(file_cache) - set(updated_cache)
    if not docstore.is_connected():
        log.error("Database connection is closed unexpectedly.")
        return  # Or handle appropriately
    keys_to_delete = [doc_id for f in deleted_files for doc_id in docstore.get_doc_ids_by_filename(f)]
    for doc_id in keys_to_delete:
        docstore.delete(doc_id)

    if keys_to_delete:
        log.info(f"Removed {len(keys_to_delete)} deleted doc(s)")

    return deleted_files


#class SQLiteVectorStoreNoPersist(SQLiteVectorStore):
#    """SQLite Vector Store that disables JSON persistence."""
#    def persist(self, *args, **kwargs) -> None:
#        pass

class SQLiteVectorStoreNoPersist(SQLiteVectorStore):
    """SQLite Vector Store that disables JSON persistence."""

    def persist(self, persist_path: Optional[str] = None, fs: Optional[Any] = None) -> None:
        log.debug("Skipping VectorStore persist (NoPersist mode).")

    def to_dict(self) -> dict:
        return {}  # Prevent image__vector_store.json creation

log = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="indexer.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=dedent("""\
            Build or update a local SQLite‑backed index for RAG search.
        """),
        epilog="Example: python3 indexer.py --dir /path/to/dir"
    )

    # — Embedding / chunking
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Sentence‑transformer model used for embeddings")
    parser.add_argument("--chunk-size", type=int, default=512,
                        metavar="TOKENS", help="Chunk size passed to node parser")
    parser.add_argument("--chunk-overlap", type=int, default=100,
                        metavar="TOKENS", help="Overlap between consecutive chunks")

    # — File system
    parser.add_argument("--dir", default=".",
                        metavar="PATH", help="Root directory to index")

    # — House‑keeping flags
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rebuild", action="store_true",
                       help="Ignore existing index; rebuild from scratch")
    group.add_argument("--purge",   action="store_true",
                       help="Delete index & cache, then exit")

    # — Misc
    parser.add_argument("--debug", action="store_true", help="Verbose logging")

    return parser.parse_args()

def load_file_cache(cache_path: Path) -> Dict[str, float]:
    """Load cached file metadata to track modified files."""
    try:
        with cache_path.open("r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        log.warning(f"Failed to load cache: {e}")
        return {}

def load_new_documents(root_dir: Path, file_cache: Dict[str, float]) -> Tuple[List[Document], Dict[str, float]]:
    """
    Load new or modified documents from a given directory.

    Args:
        root_dir (Path): Directory to scan for new files.
        file_cache (Dict[str, float]): Cache of previously processed files.

    Returns:
        Tuple[List[Document], Dict[str, float]]: New documents and updated file cache.
    """
    new_docs = []
    updated_cache = {}

    for filepath in root_dir.rglob("*"):
        if not filepath.is_file() or filepath.suffix.lower() not in Config.supported_extensions or filepath.name == Config.cache_path.name:
            continue

        mtime = filepath.stat().st_mtime
        filepath_str = str(filepath.resolve())
        updated_cache[filepath_str] = mtime

        if file_cache.get(filepath_str) == mtime:
            log.debug(f"Unchanged: {filepath.relative_to(root_dir)}")
            continue

        try:
            with filepath.open("r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            log.warning(f"Could not decode: {filepath}")
            continue

        # Create Document instead of Node
        new_docs.append(Document(text=f"Filename: {filepath}\n\n{content}", metadata={"filename": filepath_str, "keywords": extract_keywords(content)}))
        log.debug(f"Indexed: {filepath.relative_to(root_dir)}")

    log.info(f"Discovered {len(new_docs)} new/modified file(s)")
    return new_docs, updated_cache

def extract_keywords(text: str, num_keywords: int = 5) -> list:
    """Extract keywords from text based on word frequency."""
    words = re.findall(r'\w+', text.lower())  # Extract words, convert to lowercase
    word_counts = Counter(words)
    most_common = word_counts.most_common(num_keywords)
    return [word for word, _ in most_common]

def main() -> None:
    """
    Entry point for the RAG Indexer.

    This function performs the indexing process, including file scanning, document parsing,
    index creation, and persistence. It supports rebuilding and purging of existing indices.
    """
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    args = parse_args()

    parser = SimpleNodeParser(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    configure_settings(args.model, parser)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Handle purging
    if args.purge:
        log.warning("Purging index and cache...")

        if Config.index_dir.exists():
            if not os.access(Config.index_dir, os.W_OK):
                log.error(f"No write permission to delete {Config.index_dir}. Try: sudo chown -R $USER:$USER {Config.index_dir}")
                sys.exit(1)
            shutil.rmtree(Config.index_dir)

        if Config.cache_path.exists():
            Config.cache_path.unlink()

        log.info("Purge completed. Skipping indexing process.")
        log.info("To rebuild the index, run again without --purge or with --rebuild.")
        return

    # If the index folder is missing but cache exists, remove the stale cache
    elif not Config.index_dir.exists() and Config.cache_path.exists():
        log.warning(f"Index folder missing but cache exists. Removing stale cache.")
        Config.cache_path.unlink()

    log.info(f"Embedding model: {args.model}")
    log.info(f"Index directory: {Config.index_dir}")
    log.info(f"Chunking with size={args.chunk_size} overlap={args.chunk_overlap}")

    Config.index_dir.mkdir(parents=True, exist_ok=True)

    # Load file cache
    file_cache = load_file_cache(Config.cache_path)
    log.info(f"Loaded file cache with {len(file_cache)} entries")

    new_nodes = []; deleted_files = set(); updated_cache = {}; total_tokens = 0
    start = time.perf_counter()

    # Load new documents unless purging (which already exited)
    if args.rebuild or not args.purge:
        new_docs, updated_cache = load_new_documents(Path(args.dir), file_cache)
        log.info(f"Scanned {len(updated_cache)} files")
        new_nodes = parser.get_nodes_from_documents(new_docs)

        kvstore = SQLiteKVStore(str(Config.sqlite_path))
        kvstore["total_files_seen"] = len(updated_cache)
        kvstore["index_runs"] = kvstore.get("index_runs", 0) + 1
        kvstore["last_index_run"] = time.time()
        log.info(f"Last index run: {time.ctime(kvstore['last_index_run'])}")

        vector_store = SQLiteVectorStoreNoPersist(str(Config.sqlite_path))
        with SQLiteDocStore(str(Config.sqlite_path)) as docstore:
            storage_context = StorageContext.from_defaults(
                docstore=docstore,
                index_store=SQLiteIndexStore(str(Config.sqlite_path)),
                vector_store=vector_store,
                graph_store=SQLiteGraphStore(str(Config.sqlite_path)),
                persist_dir=None,
            )
            storage_context.kvstore = kvstore

            assert hasattr(docstore, 'get_all_docs'), "Custom SQLiteDocStore is not being used!"
            log.debug(f"KVStore 'last_index_run': {storage_context.kvstore['last_index_run']}")
            log.debug(f"Docstore backend: {type(storage_context.docstore)}")
            log.debug(f"Vectorstore backend: {type(storage_context.vector_store)}")
            log.debug(f"Indexstore backend: {type(storage_context.index_store)}")

            try:
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    storage_context=storage_context,
                )
                log.info("Loaded existing index from vector store.")
            except Exception as e:
                log.warning(f"Index not found or load failed: {e}")
                log.warning("Creating new index with new nodes.")
                index = VectorStoreIndex.from_documents(
                    new_docs,
                    storage_context=storage_context,
                    store_text=True,
                )

            docstore.add_documents(new_docs)
            storage_context.index_store.add_index_struct(index.index_struct)
            storage_context.index_store.persist()        # commits the change
            all_docs = docstore.get_all_docs()
            kvstore["total_docs"] = len(all_docs)

            total_tokens = sum(
                len(doc.text.split()) for doc in all_docs.values()
                if isinstance(doc, Document) and doc.text
            )
            log.info(f"Total tokens in index: {total_tokens}")

            kvstore["docs_indexed"] = len(new_docs)
            kvstore["tokens_indexed"] = total_tokens

            for doc_id, doc in all_docs.items():
                log.debug(f"Doc ID: {doc_id} | Type: {type(doc)} | Text Exists: {hasattr(doc, 'text')} | Text Len: {len(doc.text.split()) if hasattr(doc, 'text') and doc.text else 'NO TEXT'}")

            # Perform necessary operations
            if new_nodes:
                index.insert_nodes(new_nodes)
                log.info(f"Inserted {len(new_nodes)} new nodes.")
            else:
                log.info("No new nodes to insert.")

            # Perform cleanup for deleted files inside the `with` block
            deleted_files = cleanup_deleted(docstore, file_cache, updated_cache)
            kvstore["deleted_files"] = len(deleted_files) if deleted_files else 0

            # Only persist what you want explicitly
            docstore.persist()
            storage_context.index_store.persist()
            storage_context.graph_store.persist()

            # Don't close the docstore connection until after all necessary operations are completed
            with Config.cache_path.open("w") as f:
                json.dump(updated_cache, f, indent=2)

            if not docstore.check_index_consistency():
                log.error("Index inconsistency detected!")
                return

            # Log the successful completion
            log.info(f"Index updated at {Config.index_dir}")

    elapsed = time.perf_counter() - start
    kvstore["last_run_duration"] = elapsed

    log.info("\n--- Indexing Summary ---\n"
            f" 📝 New nodes inserted: {len(new_nodes) if new_nodes else 0}\n"
            f" ❌ Deleted files     : {len(deleted_files) if deleted_files else 0}\n"
            f" 📚 Total docs indexed: {len(updated_cache)}\n"
            f" 🔢 Total tokens      : {total_tokens}\n"
            f" ⏱️  Elapsed time      : {elapsed:.2f}s")

    log.info("\n--- KVStore Stats ---")
    for key in ["index_runs", "last_index_run", "docs_indexed", "tokens_indexed", "last_run_duration", "deleted_files", "total_files_seen", "total_docs"]:
        log.info(f"📦 {key}: {kvstore.get(key)}")

if __name__ == "__main__":
    main()
