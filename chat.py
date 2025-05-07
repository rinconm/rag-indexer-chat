# Retrieve-Augmented Generator (RAG) Chat Interface

# Standard Library
import sys
import time
import json
import signal
import logging
import argparse
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
from difflib import get_close_matches
from textwrap import shorten

# Third-party
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown
from rich.logging import RichHandler
from llama_index.core import StorageContext, Settings, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
from llama_index.core.schema import Node
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from transformers import GPT2TokenizerFast

# Local SQLite Storage
from backend.sqlite_docstore import SQLiteDocStore
from backend.sqlite_indexstore import SQLiteIndexStore
from backend.sqlite_vectorstore import SQLiteVectorStore


console = Console()
log = logging.getLogger(__name__)
FILENAME_RE = re.compile(r"\b[\w\-/]+\.\w+\b")

@dataclass(frozen=True)
class Config:
    VERSION = "1.0.0"
    index_dir: Path = Path("./index")
    sqlite_path: Path = index_dir / "sqlite.db"
    chatlog_dir  : Path = Path("./chat_logs")
    system_prompt = "You are a helpful assistant answering questions about local project files concisely and accurately."

class HybridRetriever(VectorIndexRetriever):
    """
    Custom retriever supporting both keyword filter and embedding-based search.
    """
    def retrieve(self, query_str: str) -> List[Node]:
        log.debug(f"HybridRetriever running query: '{query_str}'")
        log.debug(f"Filters used: {[k for k in query_str.lower().split()]}")

        embedding = Settings.embed_model.get_text_embedding(query_str)

        query = VectorStoreQuery(
            query_str=query_str,
            query_embedding=embedding,
            mode=VectorStoreQueryMode.DEFAULT,
            #filters=MetadataFilters(filters=[MetadataFilter(key="keywords", value=k) for k in query_str.lower().split()]),
        )
        results = self._index.vector_store.query(query)
        log.debug(f"Scores: {[r.score for r in results.nodes or []]}")

        if results.nodes and max(r.score for r in results.nodes) < 0.3:
            log.warning("All retrieved scores are low. Consider tweaking your index or query.")

        if not results.nodes:
            log.debug("Trying fallback: pure embedding query")
            results = self._index.vector_store.query(
                VectorStoreQuery(query_str=query_str,
                query_embedding=embedding,
                mode=VectorStoreQueryMode.DEFAULT)
           )

        return [self._node_postprocess(r) for r in results.nodes or []]

def configure_logging(debug: bool):
    """Configure the logging behavior."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)]
    )

def configure_settings(llm: str, embed: str):
    """Configure LLM and embedding settings."""
    Settings.store_text = True
    Settings.embed_model = HuggingFaceEmbedding(model_name=embed)
    Settings.llm = Ollama(model=llm)

def build_metadata(args: argparse.Namespace, history: List[Tuple[str, str]]) -> dict:
    """Build metadata for saving chat history."""
    return {
        "llm": args.llm,
        "embedding": args.embed,
        "retrieved_top_k": args.top_k,
        "history_len": len(history),
    }

def load_storage() -> Tuple[StorageContext, Any]:
    storage = StorageContext.from_defaults(
        docstore=SQLiteDocStore(str(Config.sqlite_path)),
        index_store=SQLiteIndexStore(str(Config.sqlite_path)),
        vector_store=SQLiteVectorStore(str(Config.sqlite_path)),
    )

    index_structs = storage.index_store.get_index_structs_dict()
    if not index_structs:
        log.warning("Index store empty – reconstructing from vector store")
        index = VectorStoreIndex.from_vector_store(           # ← instant rebuild
            vector_store=storage.vector_store,
            storage_context=storage,
        )
        storage.index_store.add_index_struct(index.index_struct)
        storage.index_store.persist()
        return storage, index

    index_id = next(iter(index_structs))
    return storage, storage.index_store.get_index(index_id)

def filename_in_query(q: str, files: set[str]) -> str | None:
    for token in FILENAME_RE.findall(q):
        if token in files:
            return token
        return None

def get_nodes(user_input: str, filenames: set[str], indexed_nodes: List[Node], retriever: HybridRetriever) -> List[Node]:
    fname = filename_in_query(user_input, filenames)
    if fname:
        nodes = [n for n in indexed_nodes
                if Path(n.metadata.get("filename", "unknown")).name == fname]
        if log.level == logging.DEBUG:
            console.print(f":page_facing_up: [cyan]Using file match:[/] {fname}")
        return nodes

    """Retrieve nodes from either filename match or search."""
    if user_input.strip() in filenames:
        nodes = [node for node in indexed_nodes if Path(node.metadata.get("filename", "unknown")).name == user_input.strip()]
        console.print(f":file_folder: [cyan]Matched file:[/] {user_input.strip()} — {len(nodes)} chunk(s)")
    else:
        nodes = retriever.retrieve(user_input)
    return nodes

def format_history(history: List[Tuple[str, str]]) -> str:
    """Format chat history for display."""
    return "\n".join(f"User: {q}\nAI: {a}" for q, a in history)

def extract_filenames(nodes: List[Node]) -> List[str]:
    """Extract unique filenames from a list of nodes."""
    return sorted({Path(node.metadata.get("filename", "unknown")).name for node in nodes})

def chat_loop(index: Any, retriever: HybridRetriever, system_prompt: str, args: argparse.Namespace, filenames: set[str], indexed_nodes: List[Node]) -> None:
    """
    Main chat loop to interact with the user, process input, retrieve context, generate responses, and manage chat history.

    Args:
        index (Any): The loaded index for retrieval.
        retriever (HybridRetriever): The custom retriever for hybrid search.
        system_prompt (str): The system prompt that guides the behavior of the AI.
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        None
    """
    history: List[Tuple[str, str]] = []
    chatlog_updated = False  # track save status
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    try:
        while True:
            user_input = Prompt.ask("[bold magenta]You")
            if user_input.lower().strip() in {"exit", "quit"}:
                console.print("\n[green]Goodbye! Exiting...")
                break

            if not user_input:
                continue  # empty input = skip query, show prompt again

            match user_input.strip().lower():
                case "/help":
                    console.print(Panel("""Available commands:
                /files    - List all indexed files
                /meta     - Show current config info
                /debug    - Toggle debug logging
                /history  - Show recent conversation turns
                /reset    - Clear chat history
                /save     - Save chat log now
                /tokens   - Estimate current token usage
                /help     - Show this help
                exit      - Quit chat
                """, title="[bold blue]Commands"))

                case _ if user_input.startswith("/search "):
                    partial = user_input[len("/search "):].strip()
                    if not partial:
                        console.print("[yellow]Usage:[/] /search <partial_filename>")
                        continue
                    matches = get_close_matches(partial, filenames, n=5, cutoff=0.4)
                    console.print(f"[cyan]Search results:[/] {', '.join(matches) or 'none'}")

                case "/files":
                    console.print(f"[cyan]Files:[/] {', '.join(sorted(filenames))}")

                case "/debug":
                    log.setLevel(logging.DEBUG if log.level != logging.DEBUG else logging.INFO)
                    new_level = "ON" if log.level == logging.DEBUG else "OFF"
                    console.print(f"[yellow]Debug mode toggled to:[/] {new_level}")

                case "/meta":
                    console.print(Panel(f"""
                    [cyan]Current Configuration:[/]
                    - LLM: {args.llm}
                    - Embedding: {args.embed}
                    - Top K: {args.top_k}
                    - Chatlog: {args.chatlog}
                    - Debug Mode: {"ON" if log.level == logging.DEBUG else "OFF"}
                    """, title="[bold blue]Configuration"))

                case "/reset", "/clear":
                    history.clear()
                    console.print("[yellow]Chat history cleared.[/]")

                case "/save":
                    if chatlog_updated:
                        args.chatlog.write_text(json.dumps({"history": history, "meta": build_metadata(args, history)}, indent=2))
                        console.print(f"[yellow]Chat history saved to:[/] {args.chatlog}")
                    else:
                        console.print("[red]No chatlog path configured.[/]")

                case "/tokens":
                    estimated_tokens = sum(len(tokenizer.encode(t)) for h in history for t in h) + len(tokenizer.encode(user_input))
                    console.print(f"[cyan]Estimated tokens in current history/context:[/] ~{estimated_tokens}")

                case "/history":
                    console.print(Panel("\n".join(f"[cyan]{i+1}. {q}[/]\n{a}" for i, (q, a) in enumerate(history)), title="[bold blue]Chat History"))

                case "/export":
                    export_path = Path(f"export_{time.strftime('%Y%m%d_%H%M%S')}.json")
                    export_path.write_text(json.dumps([n.to_dict() for n in indexed_nodes], indent=2))
                    console.print(f"[yellow]Exported all indexed nodes to:[/] {export_path}")

                case "hello there", "general kenobi":
                    console.print("[blue]General Kenobi![/] :robot_face:")

                case "/vibecheck":
                    console.print(":sparkles: [green]Vibe is immaculate[/] :sparkles:")

            # Retrieve nodes based on user input
            start = time.perf_counter()
            nodes = get_nodes(user_input, filenames, indexed_nodes, retriever)
            if not nodes:
                if log.level == logging.DEBUG:
                    console.print("[yellow]No relevant context found. Proceeding without context.[/]")
            elapsed = time.perf_counter() - start
            log.debug(f"Retrieved {len(nodes)} nodes in {elapsed:.2f}s")

            retrieved_filenames = extract_filenames(nodes)            
            if log.level == logging.DEBUG:
                console.print(
                    f"[green]Retrieved {len(nodes)} chunk(s)"
                    f" from {len(retrieved_filenames)} file(s): "
                    f"{', '.join(retrieved_filenames) or 'none'} in {elapsed:.2f}s"
                )

            matches = get_close_matches(user_input.strip(), filenames, n=1, cutoff=0.8)

            if not nodes and matches:
                console.print(f"[yellow]Did you mean:[/] {matches[0]} ?")

            snippets = []
            for i, node in enumerate(nodes, 1):
                snippet = node.text.strip().replace("\n", " ")[:160]
                meta = node.metadata or {}
                filename = Path(node.metadata.get("filename", "unknown"))
                display_name = str(filename) if args.verbose else filename.name
                snippets.append(f"[blue]Chunk {i} ({display_name}):[/] {snippet}...")

            if log.level == logging.DEBUG:
                console.print(Panel("\n".join(snippets), title="[bold cyan]Retrieved Context", expand=False))

            console.rule()

            context = "\n\n".join(node.text for node in nodes)
            context = shorten(context, width=16000, placeholder="\n\n...[truncated]...")
            log.debug("Context word count: %d", len(context.split()))
            history_block = format_history(history)
            final_prompt = f"{system_prompt}\n\nContext:\n{context}\n\n{history_block}\nUser: {user_input}\nAI:"

            # Generate response using LLM
            start = time.perf_counter()
            raw_response = Settings.llm.complete(final_prompt, stream=False)
            if not raw_response or not hasattr(raw_response, "text"):
                console.print("[red]LLM failed to generate response.[/]")
                continue
            elapsed = time.perf_counter() - start

            response = raw_response.text.strip()

            if not response:
                console.print("[yellow]No response generated.[/]")
                continue

            log.info("Response generated in %.2fs | %d tokens", elapsed, len(response.split()))
            console.rule()

            response = response.replace("```", "\u200b```")
            console.print(Panel(Markdown(response), title="[cyan]AI Response"))

            history.append((user_input, response))
            history = history[-args.history_limit:]

            # Save chat history after successful turn
            if args.chatlog:
                args.chatlog.write_text(json.dumps({"history": history, "meta": build_metadata(args, history)}, indent=2))
                if log.level == logging.DEBUG:
                    console.print(f"[yellow]Chat history auto-saved to:[/] {args.chatlog}")
                chatlog_updated = True

    except KeyboardInterrupt:
        console.print("\n[red]Interrupted by user.")

    except Exception as e:
        log.exception(f"Unhandled exception: {e}")

    finally:
        # Save chat history on exit or crash
        if args.chatlog and history:
            args.chatlog.write_text(json.dumps({"history": history, "meta": build_metadata(args, history)}, indent=2))
            console.print(f"[yellow]Final chat history auto-saved to:[/] {args.chatlog}")
        console.print()
        console.print("[green]Thanks for chatting! See you next time :wave:")

def count_unique_files(nodes: List[Node]) -> int:
    return len({node.metadata.get('filename', 'unknown') for node in nodes})

def main():
    """
    Entry point for RAG Chat CLI.

    Parses command-line arguments, configures settings, 
    loads index storage, and starts the interactive chat loop.
    """
    parser = argparse.ArgumentParser(
        description="Chat with your local indexed files using RAG.",
        epilog="Example: python chat.py --llm mistral --embed all-MiniLM-L6-v2 --chatlog chat.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--llm", type=str, default="mistral", help="LLM model for answering")
    parser.add_argument("--embed", type=str, default="all-MiniLM-L6-v2", help="Embedding model for retrieval")
    parser.add_argument("--top-k", type=int, default=10, help="Top K retrieved chunks")
    parser.add_argument("--chatlog", type=Path, help="Optional file to save chat history")
    parser.add_argument("--history-limit", type=int, default=20, help="Max number of turns to keep in chat history")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--verbose", action="store_true", help="Show full paths in retrieved context")
    parser.add_argument("--version", action="version", version=f"RAG Chat {Config.VERSION}")

    args = parser.parse_args()

    configure_logging(args.debug)

    if not Config.index_dir.exists():
        console.print(f"[red]Error:[/] No index found at {Config.index_dir}.")
        console.print("[yellow]Hint:[/] Run: python indexer.py --dir /path/to/code")
        sys.exit(1)

    if args.embed.lower() == args.llm.lower():
        console.print(f"[bold red]Warning:[/] LLM and Embedding model appear to be the same. Are you sure?")

    if not args.debug:
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)

        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()

    configure_settings(llm=args.llm, embed=args.embed)

    Config.chatlog_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel("[bold cyan]Retrieve-Augmented Generator Chat[/]\n[dim]powered by LlamaIndex, Ollama, HuggingFace", style="bold green"))

    if not args.chatlog:
        args.chatlog = (Config.chatlog_dir /
                        f"chat_{time.strftime('%Y%m%d_%H%M%S')}.json").resolve()

    system_prompt = Config.system_prompt

    storage, index = load_storage()

    log.debug(f"Index structs: {len(storage.index_store.get_index_structs_dict())}")
    indexed_nodes = list(storage.docstore.get_all_docs().values())
    filenames = {Path(node.metadata.get("filename", "unknown")).name for node in indexed_nodes}

    retriever = HybridRetriever(
        index=index,
        similarity_top_k=args.top_k,
    )
    if not hasattr(index, "as_retriever"):
        raise ValueError("Loaded index does not support retrieval.")

    doc_count   = len(indexed_nodes)
    file_count  = count_unique_files(indexed_nodes)

    startup_panel = Panel(
        "\n".join([
            f"[bold cyan]LLM:[/] {args.llm}   "
            f"[bold cyan]Embeddings:[/] {args.embed}   "
            f"[bold cyan]Top K:[/] {args.top_k}   "
            f"[bold cyan]Docs:[/] {doc_count}   "
            f"[bold cyan]Files:[/] {file_count}"
            "\n",
            "[bold green]Chat ready!  Type [bold cyan]exit[/] to quit or [bold cyan]/help[/] for commands.",
        ]),
        title="[bold blue]RAG Chat Startup",
    )
    console.print(startup_panel)

    chat_loop(index, retriever, system_prompt, args, filenames, indexed_nodes)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    main()
