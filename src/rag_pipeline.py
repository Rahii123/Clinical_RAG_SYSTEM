import os
import re
from typing import List
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
import requests

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


try:
    from src.embeddings import OpenAIEmbeddingFunction
except ImportError:
    from embeddings import OpenAIEmbeddingFunction

embeddings = OpenAIEmbeddingFunction()
# ==============================
# CONFIG
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSIST_DIRECTORY = os.path.join(BASE_DIR, ".vector_store")

GROQ_MODEL = "llama-3.3-70b-versatile"

TOP_K = 8
MMR_FETCH_K = 30
MMR_LAMBDA = 0.5





# ==============================
# LOAD VECTOR DB
# ==============================

def load_vector_db():

    vectordb = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name="clinical_guidelines"
    )

    count = vectordb._collection.count()
    print(f"\n📊 Collection: clinical_guidelines | Chunks: {count}")

    if count == 0:
        print("❌ Vector store empty. Run chroma_store.py")
        return None

    return vectordb


# ==============================
# RETRIEVAL (Transparent)
# ==============================

def retrieve_documents(query: str, vectordb) -> List[Document]:
    print("\n📚 Searching with MMR...")
    # Query expansion/clarification placeholder
    if len(query.strip().split()) < 4:
        # For short queries, expand using LLM or prompt user (placeholder)
        print("⚠ Query is short or vague. Consider expanding or clarifying.")
        # Optionally, call LLM to rewrite query here

    # Step 1 — Get diverse documents (MMR)
    mmr_docs = vectordb.max_marginal_relevance_search(
        query,
        k=TOP_K,
        fetch_k=MMR_FETCH_K,
        lambda_mult=MMR_LAMBDA
    )

    # Step 2 — Get similarity scores separately
    scored_docs = vectordb.similarity_search_with_score(query, k=TOP_K)

    # Create score lookup
    score_lookup = {}
    for doc, score in scored_docs:
        score_lookup[doc.page_content[:200]] = score

    print("\n🔎 Retrieval Transparency Report")
    print("-" * 60)

    final_docs = []
    seen = set()

    for i, doc in enumerate(mmr_docs, 1):
        key = doc.page_content[:200]
        if key in seen:
            continue
        seen.add(key)
        score = score_lookup.get(key, "N/A")
        source = doc.metadata.get("source", "Unknown")
        print(f"[{i}] Score: {score} | Source: {source}")
        print(f"     Preview: {doc.page_content[:120]}...\n")
        final_docs.append(doc)

    print("-" * 60)

    return final_docs


# ==============================
# PROMPT WITH STRICT CITATION
# ==============================

def get_prompt():
    template = """
You are a clinical AI assistant.

Strict Rules:
1. Use ONLY the provided sources below.
2. After EVERY factual claim, cite like this: [Source X]
3. If information is missing, write: Not in guidelines.
4. Do NOT invent, infer, summarize, or add any information not explicitly present in the context. If unsure, say 'Not in guidelines.'
5. Every sentence MUST be followed by a citation. If you cannot cite, do not include the sentence.

Context:
{context}

Question:
{question}

Instructions:
Answer ONLY the user's question directly. Do NOT include definitions, criteria, or management sections unless specifically asked. Do NOT add any extra sections or information not requested by the user.
"""
    return ChatPromptTemplate.from_template(template)


# ==============================
# GENERATE ANSWER
# ==============================

def generate_answer(question: str, context: str):
    llm = ChatGroq(
        temperature=0.1,
        model=GROQ_MODEL
    )
    prompt = get_prompt()
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })
    # Post-process: Remove any sentence not followed by a citation
    import re
    answer = response.content
    # Split into sentences (simple split, can be improved)
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    filtered = [s for s in sentences if re.search(CITATION_PATTERN, s)]
    # Remove section headers unless specifically asked
    # Only keep cited sentences
    final = list(dict.fromkeys(filtered))
    return '\n'.join(final)


# ==============================
# REGEX VALIDATION LAYER
# ==============================

REQUIRED_SECTIONS = [
    r"\*\*Definition\*\*",
    r"\*\*Criteria\*\*",
    r"\*\*Management\*\*"
]

CITATION_PATTERN = r"\[Source\s+\d+\]"

def validate_answer(answer: str):
    print("\n📋 Validation Report")
    print("-" * 40)

    # Check structure
    missing_sections = []
    for pattern in REQUIRED_SECTIONS:
        if not re.search(pattern, answer):
            missing_sections.append(pattern)

    if missing_sections:
        print("⚠ Missing Sections:", missing_sections)
    else:
        print("✅ All required sections present")

    # Check citations
    citations = re.findall(CITATION_PATTERN, answer)

    if len(citations) == 0:
        print("⚠ No citations detected")
    else:
        print(f"✅ Citations detected: {len(citations)}")

    print("-" * 40)
def build_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content.strip()
        # Safety truncation (avoid token explosion)
        if len(content) > 1200:
            content = content[:1200] + "\n...[truncated]"
        # Only [Source {i}] to match validation regex
        parts.append(
            f"[Source {i}]\n"
            f"{content}\n"
            f"{'─'*70}"
        )
    return "\n\n".join(parts)


# ==============================
# MAIN LOOP
# ==============================

def main():
    vectordb = load_vector_db()
    if not vectordb:
        return

    print(f"\n🩺 Clinical RAG Ready | {GROQ_MODEL}\n")

    while True:
        query = input("🩺 Query (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        docs = retrieve_documents(query, vectordb)

        if not docs:
            print("❌ No relevant documents found.")
            continue

        context = build_context(docs)
        answer = generate_answer(query, context)

        print("\n" + "=" * 70)
        print("📋 CLINICAL ANSWER")
        print("=" * 70)
        print(answer)
        print("=" * 70)

        validate_answer(answer)


if __name__ == "__main__":
    main()