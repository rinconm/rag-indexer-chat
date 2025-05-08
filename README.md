# RAG Indexer + Chat
A fast, local-first Retrieve-Augmented Generation (RAG) indexing system built using LlamaIndex and SQLite. Index your project files, power intelligent search, and run local chat with local LLMs (default Ollama).

---

## How it works (high‑level)
1. indexer.py walks the target directory, splits files into chunks, embeds
them with a HuggingFace model, and stores everything in index/sqlite.db
via custom SQLite back‑ends.

2. chat.py loads the index, employs a hybrid retriever (embedding + keyword
filter), builds a prompt with retrieved context, and streams the answer from
your local LLM.

---

## Directory Structure
```bash
rag-indexer-chat/
├── backend/                  # all reusable SQLite helpers
│   ├── __init__.py
│   ├── sqlite_kvstore.py     # key‑value store
│   ├── sqlite_docstore.py    # docstore
│   ├── sqlite_indexstore.py  # index meta
│   ├── sqlite_vectorstore.py # vectors & embeddings
│   └── sqlite_graphstore.py  # (optional) graph support
├── indexer.py                # CLI to create / update index
├── chat.py                   # interactive chat CLI
├── index/                    # runtime artefacts (auto‑created)
│   ├── sqlite.db             # single SQLite database
│   └── .index_cache.json     # file‑modified‑time cache
├── chat_logs/                # auto‑saved chat histories
├── README.md                 # you are here
└── requirements.txt          # minimal deps
```

---

## Getting Started
### 1. Clone the repository

```bash
git clone https://github.com/rinconm/rag-indexer-chat.git
cd rag-indexer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start indexing a directory

```bash
python3 indexer.py --dir /path/to/your/files
```

### 4. Chat with your data

```bash
python3 chat.py
```

---

## Sample Output
### 1. Indexer Help Menu
```bash
mrincon@DESKTOP-JIRIPVL:~/scripts/rag-indexer-chat$ python3 indexer.py -h
usage: indexer.py [-h] [--model MODEL] [--chunk-size TOKENS] [--chunk-overlap TOKENS] [--dir PATH] [--rebuild | --purge] [--debug]

Build or update a local SQLite‑backed index for RAG search.

options:
  -h, --help            show this help message and exit
  --model MODEL         Sentence‑transformer model used for embeddings (default: all-MiniLM-L6-v2)
  --chunk-size TOKENS   Chunk size passed to node parser (default: 512)
  --chunk-overlap TOKENS
                        Overlap between consecutive chunks (default: 100)
  --dir PATH            Root directory to index (default: .)
  --rebuild             Ignore existing index; rebuild from scratch (default: False)
  --purge               Delete index & cache, then exit (default: False)
  --debug               Verbose logging (default: False)

Example: python3 indexer.py --dir /path/to/dir
```

### 2. Chat Help Menu
```bash
mrincon@DESKTOP-JIRIPVL:~/scripts/rag-indexer-chat$ python3 chat.py -h
usage: chat.py [-h] [--llm LLM] [--embed EMBED] [--top-k N] [--chatlog CHATLOG] [--history-limit TURNS] [--verbose] [--debug] [--version]

Interactive RAG chat over your local index

options:
  -h, --help            show this help message and exit
  --llm LLM             LLM served by Ollama for final answers (default: mistral)
  --embed EMBED         Sentence‑transformer model for embeddings (default: all-MiniLM-L6-v2)
  --top-k N             How many chunks to retrieve per query (default: 10)
  --chatlog CHATLOG     Path to save chat log JSON (auto‑generated if omitted) (default: None)
  --history-limit TURNS
                        Max QA pairs kept in rolling context (default: 20)
  --verbose             Show full file paths in results (default: False)
  --debug               Verbose logging (default: False)
  --version             show program's version number and exit
```

### 3. Indexing Example

![image](https://github.com/user-attachments/assets/b3319965-a972-4f76-8d25-22478f26e6f5)

### 4. Chat Example

![image](https://github.com/user-attachments/assets/9be35bb6-66e9-408c-bde6-6eeaeaa4223a)

---

## Configuration

- Modify `Config` in `indexer.py` and 'chat.py' to customize paths and supported file types.
- Embedding model defaults to `all-MiniLM-L6-v2`.
- LLM defaults to `Ollama` with `mistral` (you must have Ollama installed and running).

---

## Supported File Types

- `.py`, `.sh`, `.md`, `.txt`, `.yaml`, `.json`, `.html`, `.js`, `.ts`, `.css`, `.java`, `.cpp`, `.go`, etc.

---

## License

MIT License. See [LICENSE](./LICENSE) for details.

---

## Note on Authorship

This project was built with the help of AI-assisted tools like GitHub Copilot and OpenAI's ChatGPT. Contributions welcome!
