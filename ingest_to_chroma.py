import argparse
import os
import re
import uuid
from pathlib import Path
from typing import List, Dict, Iterable, Tuple

import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader


# --------- Utilities ---------

def load_text_from_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        texts = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt:
                texts.append(txt)
        return "\n".join(texts).strip()
    except Exception as e:
        print(f"[WARN] Failed to parse PDF: {path} ({e})")
        return ""


def load_text_from_plain(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception as e:
        print(f"[WARN] Failed to read text: {path} ({e})")
        return ""


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    by_sentence: bool = True
) -> List[str]:
    """
    Recursive-ish chunker:
      - If by_sentence=True, tokenizes into sentences and packs them until ~chunk_size chars.
      - Overlaps the last ~chunk_overlap chars between chunks for context continuity.
    """
    if not text:
        return []

    if by_sentence:
        sents = [normalize_ws(s) for s in sent_tokenize(text)]
        chunks = []
        buf = ""
        for s in sents:
            if not s:
                continue
            if len(buf) + len(s) + 1 <= chunk_size:
                buf = (buf + " " + s).strip() if buf else s
            else:
                if buf:
                    chunks.append(buf)
                    # create overlap window
                    if chunk_overlap > 0 and len(buf) > chunk_overlap:
                        buf = buf[-chunk_overlap:]
                    else:
                        buf = ""
                # start new buffer with current sentence
                if len(s) > chunk_size:
                    # hard-split very long sentence
                    for i in range(0, len(s), chunk_size):
                        part = s[i:i+chunk_size]
                        if part:
                            chunks.append(part)
                    buf = ""
                else:
                    buf = s
        if buf:
            chunks.append(buf)
        return chunks

    # simpler char-based fallback
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        start = end - chunk_overlap if end < n else end
        if start < 0:
            start = 0
    return [normalize_ws(c) for c in chunks if c.strip()]


def iter_docs(root: Path) -> Iterable[Tuple[str, Path]]:
    """
    Yields (doctype, path) for supported files under root.
    """
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in {".pdf", ".txt", ".md"}:
            yield ext[1:], p


def make_metadata(file_path: Path, chunk_idx: int, total_chunks: int) -> Dict:
    return {
        "source_path": str(file_path.resolve()),
        "filename": file_path.name,
        "chunk_index": chunk_idx,
        "chunk_count": total_chunks,
    }


# --------- Main ingestion ---------

def main():
    parser = argparse.ArgumentParser(description="Ingest local papers into Chroma.")
    parser.add_argument("--input_dir", required=True, type=str, help="Folder with PDFs/TXT/MD")
    parser.add_argument("--persist_dir", default="./chroma_store", type=str, help="Chroma persistence directory")
    parser.add_argument("--collection", default="papers", type=str, help="Chroma collection name")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", type=str,
                        help="SentenceTransformer model (e.g., BAAI/bge-small-en-v1.5)")
    parser.add_argument("--chunk_size", default=1200, type=int, help="Chars per chunk")
    parser.add_argument("--chunk_overlap", default=200, type=int, help="Overlap chars")
    parser.add_argument("--batch_size", default=128, type=int, help="Upsert batch size")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collection")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    assert input_dir.exists() and input_dir.is_dir(), f"Input dir not found: {input_dir}"

    client = chromadb.PersistentClient(path=args.persist_dir)

    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=args.model
    )

    if args.recreate:
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=args.collection,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}  # cosine is standard for sentence embeddings
    )

    print(f"[INFO] Using collection: {collection.name} (persist_dir={args.persist_dir})")
    print(f"[INFO] Embedding model: {args.model}")

    total_files = 0
    total_chunks = 0
    ids_batch, docs_batch, metas_batch = [], [], []

    for doctype, path in tqdm(list(iter_docs(input_dir)), desc="Scanning files"):
        total_files += 1

        if doctype == "pdf":
            text = load_text_from_pdf(path)
        else:
            text = load_text_from_plain(path)

        text = normalize_ws(text)
        if not text:
            continue

        chunks = chunk_text(
            text,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            by_sentence=True
        )
        if not chunks:
            continue

        total_chunks += len(chunks)
        for i, c in enumerate(chunks):
            # Use deterministic IDs per-file so repeated runs don't balloon the DB
            # ID = file_hash + chunk_idx would be ideal; for simplicity: filename+idx
            base = f"{path.resolve()}::{i}"
            # Make it filesystem-change resilient by hashing path+idx
            _id = str(uuid.uuid5(uuid.NAMESPACE_URL, base))
            ids_batch.append(_id)
            docs_batch.append(c)
            metas_batch.append(make_metadata(path, i, len(chunks)))

            if len(ids_batch) >= args.batch_size:
                collection.upsert(ids=ids_batch, documents=docs_batch, metadatas=metas_batch)
                ids_batch, docs_batch, metas_batch = [], [], []

    # flush remainder
    if ids_batch:
        collection.upsert(ids=ids_batch, documents=docs_batch, metadatas=metas_batch)

    print(f"[DONE] Files processed: {total_files}")
    print(f"[DONE] Chunks upserted: {total_chunks}")
    print(f"[DONE] Collection count now: {collection.count()}")
    print(f"[DONE] Persisted at: {Path(args.persist_dir).resolve()}")


if __name__ == "__main__":
    main()
