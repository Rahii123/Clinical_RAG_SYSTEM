import os
import chromadb
try:
    from src.embeddings import OpenAIEmbeddingFunction
except ImportError:
    from embeddings import OpenAIEmbeddingFunction

# ==========================================
# CHROMA SETUP
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSIST_DIRECTORY = os.path.join(BASE_DIR, ".vector_store")

client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

embedding_function = OpenAIEmbeddingFunction()

collection = client.get_collection(
    name="clinical_guidelines",
    embedding_function=embedding_function
)

# ==========================================
# QUERY
# ==========================================

query = input("Enter your question: ")

results = collection.query(
    query_texts=[query],
    n_results=5
)

print("\nTop Results:\n")

for i, doc in enumerate(results["documents"][0]):
    print(f"Result {i+1}")
    print(doc)
    print("-" * 80)