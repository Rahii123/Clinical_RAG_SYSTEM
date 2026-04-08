"""
qdrant_store.py  –  Automated Persistence Build
=========================================================
Fixed:
  ✅ INDESTRUCTIBLE Backoff Loop (Will not crash on 429).
  ✅ 60-second recovery for Free Tier stability.
  ✅ Automatic Dimension Detection (3072 support).
"""

import os
import sys
import json
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv

try:
    from src.embeddings import get_embedding_function
except ImportError:
    from embeddings import get_embedding_function

# Load environment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

# ==========================================
# CONFIG
# ==========================================

CHUNK_FOLDER = os.path.join(BASE_DIR, "data", "chunks")
STORAGE_PATH = os.path.join(BASE_DIR, "qdrant_storage")
COLLECTION_NAME = "clinical_rag_v1"
BATCH_SIZE = 15 

# ==========================================
# QDRANT SETUP
# ==========================================

client = QdrantClient(path=STORAGE_PATH)
embeddings = get_embedding_function()

# DYNAMIC DETECTION
print("🔍 Detecting embedding dimensions...")
sample = embeddings.embed_query("test")
detected_dim = len(sample)
print(f"📐 Model Dimension Detected: {detected_dim}")

# FORCE RE-CREATE
print(f"🔬 Initializing Qdrant Collection: {COLLECTION_NAME} ({detected_dim} Dimensions)")
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=detected_dim, distance=Distance.COSINE),
)

# Initialize LangChain Wrapper
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

# ==========================================
# LOAD & INDEX
# ==========================================

total_chunks = 0
file_list = sorted([f for f in os.listdir(CHUNK_FOLDER) if f.endswith(".json")])

for filename in file_list:
    filepath = os.path.join(CHUNK_FOLDER, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks: continue
    print(f"📄 Indexing: {filename} ({len(chunks)} chunks)")

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    
    # Process in batches with throttling
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_metas = metadatas[i:i + BATCH_SIZE]
        
        # SENIOR FIX: Indestructible Backoff Loop
        success = False
        retry_count = 0
        while not success:
            try:
                vector_store.add_texts(texts=batch_texts, metadatas=batch_metas)
                print(f"  ⚡ {i + len(batch_texts)}/{len(texts)} chunks uploaded...")
                time.sleep(2) # Normal throttle
                success = True
            except Exception as e:
                # Catch both the error code and the string message
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "Quota" in err_msg:
                    retry_count += 1
                    wait_time = 60 # A full minute pause to reset the quota
                    print(f"⚠️  Rate limit hit (Attempt {retry_count}). Waiting 60s for quota reset...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Unexpected Error: {e}")
                    raise e

    total_chunks += len(chunks)

print(f"\n🎯 [SUCCESS] Index Built: {COLLECTION_NAME}")
print(f"📊 Total Chunks: {total_chunks}")
