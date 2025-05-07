# RAG Indexer + Chat Interface

A local-first, privacy-conscious RAG (Retrieval-Augmented Generation) stack built with [LlamaIndex](https://github.com/jerryjliu/llama_index), SQLite, and Python. Designed for indexing directories of source code or documentation and enabling semantic chat over your data.

---

## ✨ Features

- ✅ **Indexing**: Parses and indexes a directory of text/code files.
- 🧠 **Embedding**: Uses HuggingFace embeddings with support for Mistral via Ollama.
- 💬 **Chat Interface**: Interact with your indexed data through a simple local chatbot.
- 💾 **Persistence**: Fully local storage using SQLite for documents, vectors, and index metadata.
- 📊 **KV Store**: Tracks stats and run metadata (e.g. total files, tokens, index runs).
- 🔍 **Consistency Checks**: Detect and log discrepancies in document storage/indexing.

> Built using Python 3.10+, LlamaIndex, and Ollama — with some AI help 🤖

---

## 📁 Directory Structure

```bash
.
├── indexer.py              # Main entry point for indexing a directory
├── chat.py                 # Simple command-line chatbot using indexed data
├── sqlite_docstore.py      # SQLite-backed Document Store
├── sqlite_indexstore.py    # SQLite-backed Index Store
├── sqlite_vectorstore.py   # SQLite-backed Vector Store
├── sqlite_graphstore.py    # SQLite-backed Graph Store
├── sqlite_kvstore.py       # Key-Value store for usage stats
├── requirements.txt        # Python dependencies
└── README.md               # Project overview (you are here)
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start indexing a directory

```bash
python indexer.py --dir /path/to/your/codebase
```

### 3. Chat with your data

```bash
python chat.py
```

---

## ⚙️ Configuration

- Modify `Config` in `indexer.py` to customize paths and supported file types.
- Embedding model defaults to `all-MiniLM-L6-v2`.
- LLM defaults to `Ollama` with `mistral` (you must have Ollama installed and running).

---

## 📚 Supported File Types

- `.py`, `.sh`, `.md`, `.txt`, `.yaml`, `.json`, `.html`, `.js`, `.ts`, `.css`, `.java`, `.cpp`, `.go`, etc.

---

## 📌 License

MIT License. See [LICENSE](./LICENSE) for details.

---

## 🧠 Note on Authorship

This project was built with the help of AI-assisted tools like GitHub Copilot and OpenAI's ChatGPT. Contributions welcome!
