"""
chroma_store.py  –  Build/Rebuild ChromaDB Vector Store
=========================================================
Upgrades:
  ✅ Uses OpenAI text-embedding-3-small (Ultra cost-effective)
  ✅ Persists to local .vector_store
  ✅ Progressive indexing with batching
"""

import os
import sys
import json
import time
import chromadb
from dotenv import load_dotenv

try:
    from src.embeddings import get_embedding_function
except ImportError:
    from embeddings import get_embedding_function

ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=ENV_PATH, override=True)

# ==========================================
# CONFIG
# ==========================================

CHUNK_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "chunks"))
EMBEDDING_MODEL = "text-embedding-3-small" # Highly efficient & cheap
BATCH_SIZE = 100 # OpenAI handles larger batches easily

# ==========================================
# CHROMA SETUP
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSIST_DIRECTORY = os.path.join(BASE_DIR, ".vector_store")

client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
embedding_function = get_embedding_function()

RESET = "--reset" in sys.argv
if RESET:
    try:
        client.delete_collection(name="clinical_guidelines")
        print("🗑️  Old collection deleted.")
    except Exception:
        print("ℹ️  Creating fresh collection.")

collection = client.get_or_create_collection(
    name="clinical_guidelines",
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"},
)

# ==========================================
# METADATA SANITIZER
# ==========================================

def sanitize_metadata(meta: dict) -> dict:
    clean = {}
    for k, v in meta.items():
        if isinstance(v, (str, float, bool, int)):
            clean[k] = v
        elif v is None:
            clean[k] = ""
        else:
            clean[k] = str(v)
    try:
        clean["year"] = int(clean.get("year", 0))
    except:
        clean["year"] = 0
    return clean

# ==========================================
# LOAD & INDEX
# ==========================================

total_chunks = 0
total_files  = 0

for filename in sorted(os.listdir(CHUNK_FOLDER)):
    if not filename.endswith(".json"): continue
    
    filepath = os.path.join(CHUNK_FOLDER, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks: continue
    print(f"📄 Indexing: {filename} ({len(chunks)} chunks)")

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        ids = [c["chunk_id"] for c in batch]
        docs = [c["text"] for c in batch]
        metas = [sanitize_metadata(c["metadata"]) for c in batch]

        try:
            # We pass documents; OpenAIEmbeddingFunction handles the API call
            collection.upsert(
                ids=ids,
                documents=docs,
                metadatas=metas,
            )
        except Exception as e:
            print(f"   ⚠️ Batch failed: {e}")

    total_chunks += len(chunks)
    total_files  += 1

print(f"\n🎯 Done! Chunks indexed: {collection.count()}")