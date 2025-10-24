# TinyPaperChat

## Description
A minimal, local-first Retrieval-Augmented Generation (RAG) chat service that indexes your PDFs into ChromaDB and serves a Streamlit UI backed by a LangGraph agent. It uses sentence-transformer embeddings, an optional cross-encoder reranker, and persistent Chroma storage so you can ask questions against your own library—entirely on your laptop.

---

## Components

```
[ PDFs / Notes / Papers ]
            |
            v
     Ingestion / Chunking
     (pypdf + NLTK split)
            |
            v
     Embeddings (SBERT)
            |
            v
   +---------------------+
   |  ChromaDB (persist) |
   +---------------------+
            ^
            |
  Retrieval (k, MMR, filters)
            |
   Rerank (Cross-Encoder)  (optional)
            |
            v
+--------------------+        +----------------------+
|  LangGraph Agent   | <----> |   LLM (local/API)    |
|  (tools + prompts) |        |  e.g. Ollama/OpenAI  |
+--------------------+        +----------------------+
            |
            v
      Streamlit UI
   (chat + source cites)
```

---

## Running

### 0) Prereqs
- macOS (Apple Silicon ok) or Linux
- Python 3.10+ on PATH

### 1) Install dependencies
Ensure a `requirements.txt` exists at repo root (minimal example):


### 2) Configure and run the bootstrap script

This repo expects a helper script `run_local_rag.sh` that:
- Creates a virtualenv
- Installs dependencies
- Downloads minimal NLTK tokenizers (`punkt`, `punkt_tab`)
- Creates a starter `.env`
- Starts the Streamlit app

**Edit knobs (optional):** at the top of `run_local_rag.sh`:
```bash
: "${APP_FILE:=streamlit_app.py}"                     # Streamlit entrypoint
: "${CHROMA_DB_DIR:=./data/chroma}"                   # Chroma persistence dir
: "${EMBEDDING_MODEL:=sentence-transformers/all-MiniLM-L6-v2}"
: "${RERANK_MODEL:=cross-encoder/ms-marco-MiniLM-L-6-v2}"
: "${PORT:=8501}"
```

**Make it executable and run:**
```bash
chmod +x run_local_rag.sh
./run_local_rag.sh
```

On first run, the script writes a `.env` with sensible defaults:
Open the app at: **http://localhost:8501**

### 3) Ingest your documents (once per corpus)

If you’ve already ingested your PDFs into `CHROMA_DB_DIR`, skip this step. Otherwise, use your ingestion script (e.g., `ingest_to_chroma.py`), which typically:
- Reads PDFs with `pypdf`
- Sentence-splits via NLTK
- Chunks + embeds with `sentence-transformers`
- Upserts into Chroma at `CHROMA_DB_DIR`

Run it as: 
```bash
python ingest_to_chroma.py --input_dir /path/to/papers --persist_dir /path/to/chroma/persistence/dir
```

### 4) Interact with the agent
Ask a question in the Streamlit chat. The agent will:
1. Retrieve top-k chunks from Chroma (with optional MMR)
2. Optionally rerank with a cross-encoder
3. Build a compact context with inline source handles
4. Call your configured LLM and show an answer with citations

### 5) Common tweaks

- **Change k / MMR / rerank cutoff**
  - `top_k` (e.g., 8–15)
  - `mmr_lambda` (0.5–0.8 common)
  - `rerank_top_k` (e.g., 50 → 8)

- **Switch embedding model**
  Update `EMBEDDING_MODEL` in `.env` and re-run ingestion for best results.

- **Switch generation model**
  - Local: `GEN_MODEL=ollama/llama3.1:8b-instruct`
  - Hosted: `GEN_MODEL=gpt-4o-mini` (or your choice) and set the API key.


## Folder structure (suggested)
```
.
├─ streamlit_app.py
├─ agent_graph.py
├─ retrieval_tool.py
├─ ingest_to_chroma.py
├─ requirements.txt
├─ run_local_rag.sh
├─ .env
└─ data/
   └─ chroma/        # Chroma persistence (created automatically)
```
