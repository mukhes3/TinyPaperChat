#!/usr/bin/env bash
set -euo pipefail

# ---- Config you can tweak (defaults are sane) -------------------------------
: "${PYTHON:=python3}"
: "${APP_FILE:=streamlit_app.py}"
: "${VENV_DIR:=.venv}"
: "${CHROMA_DB_DIR:=./data/chroma}"
: "${EMBEDDING_MODEL:=sentence-transformers/all-MiniLM-L6-v2}"
: "${RERANK_MODEL:=cross-encoder/ms-marco-MiniLM-L-6-v2}"
: "${PORT:=8501}"
# -----------------------------------------------------------------------------

echo "▶️  Bootstrapping Local RAG Chat…"

# 1) Ensure Python exists
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Could not find $PYTHON on PATH"; exit 1
fi

# 2) Create venv if missing
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtualenv at $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
fi

# 3) Activate venv
#   shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 4) Upgrade pip/wheel/setuptools (quiet-ish)
python -m pip install -U pip wheel setuptools >/dev/null

# 5) Install deps
if [ ! -f requirements.txt ]; then
  echo "requirements.txt not found in repo root"; exit 1
fi
echo "Installing Python dependencies (this can take a minute)…"
pip install -r requirements.txt

# 6) Minimal NLTK data (for sent_tokenize); NLTK 3.9 may want 'punkt_tab' as well
python - <<'PY'
import nltk, sys
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception as e:
            print(f"Warning: failed to download {pkg}: {e}", file=sys.stderr)
PY

# 7) Ensure Chroma persistence dir exists (won’t overwrite existing DB)
mkdir -p "$CHROMA_DB_DIR"

# 8) .env (optional): create if missing so Streamlit/LangGraph code can read it
if [ ! -f .env ]; then
  cat > .env <<EOF
# --- Runtime configuration for Local RAG Chat ---
CHROMA_DB_DIR=${CHROMA_DB_DIR}
EMBEDDING_MODEL=${EMBEDDING_MODEL}
RERANK_MODEL=${RERANK_MODEL}
# GEN_MODEL can be 'ollama/llama3.1:8b-instruct' or an API-backed model you use.
# GEN_MODEL=ollama/llama3.1:8b-instruct
# OPENAI_API_KEY=sk-...
EOF
  echo "Wrote .env with sensible defaults."
fi

# 9) Light sanity check: app file present?
if [ ! -f "$APP_FILE" ]; then
  echo "$APP_FILE not found. Make sure your Streamlit entrypoint is named correctly."; exit 1
fi

# 10) Helpful CPU setting for Mac/Unix to avoid thread thrash
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# 11) Launch Streamlit
echo "🌐 Starting Streamlit on http://localhost:${PORT} …"
exec streamlit run "$APP_FILE" --server.port "$PORT" --server.headless true
