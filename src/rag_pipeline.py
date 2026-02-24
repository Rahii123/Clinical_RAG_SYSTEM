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
SCORE_THRESHOLD = 0.5  # Lower is more similar (Cosine distance)





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
    print(f"\n📚 Searching with MMR (Threshold: {SCORE_THRESHOLD})...")
    
    # Step 1 — Get similarity scores for filtering
    # In ChromaDB with Cosine distance: 0.0 is exact match, 1.0 is opposite.
    # Score lookup for the top K documents
    scored_docs = vectordb.similarity_search_with_score(query, k=MMR_FETCH_K)
    
    # Create score lookup
    score_lookup = {doc.page_content[:200]: score for doc, score in scored_docs}

    # Step 2 — Get diverse documents (MMR)
    mmr_docs = vectordb.max_marginal_relevance_search(
        query,
        k=TOP_K,
        fetch_k=MMR_FETCH_K,
        lambda_mult=MMR_LAMBDA
    )

    print("\n🔎 Retrieval Transparency Report")
    print("-" * 60)

    final_docs = []
    seen = set()

    for i, doc in enumerate(mmr_docs, 1):
        key = doc.page_content[:200]
        if key in seen:
            continue
        
        score = score_lookup.get(key, 1.0)
        
        # Only include if it meets our quality threshold (score <= 0.5)
        if score > SCORE_THRESHOLD:
            continue

        seen.add(key)
        source = doc.metadata.get("source", "Unknown")
        print(f"[{len(final_docs)+1}] Score: {score:.4f} | Source: {source}")
        print(f"     Preview: {doc.page_content[:120]}...\n")
        final_docs.append(doc)

    print("-" * 60)

    if not final_docs:
        print("❌ No documents matched the relevance threshold.")

    return final_docs


# ==============================
# PROMPT WITH STRICT CITATION
# ==============================

def get_prompt():
    template = """
You are a Senior Clinical AI Assistant. Your goal is to provide a high-precision, safety-first answer based EXCLUSIVELY on the provided context.

### SYSTEM RULES:
1. **Acronym Definition**: At the first mention of any medical acronym (e.g., ABA, CRP, MRI), define it in parentheses.
2. **Handle Exceptions**: If the context mentions specific pathogens, age groups, or conditions that have DIFFERENT rules (e.g., Brucella requiring longer treatment), you MUST include these exceptions. Do not "cherry-pick" only the general rule.
3. **Evidence Strength**: If the context labels a recommendation as "strong", "conditional", "weak", or "high/low quality evidence", you MUST include that detail (e.g., "It is conditionally suggested that...").
4. **Synthesis**: Synthesize information from MULTIPLE sources. If Source 1 and Source 2 discuss the same topic, combine their insights rather than relying on just one.
5. **No Hallucinations**: If the information is not explicitly in the provided sources, state "Information not found in guidelines."
6. **Strict Citation**: Every single factual claim must be followed by its source in brackets, e.g., [Source 1]. Do not group citations at the end of a paragraph.

### CONTEXT:
{context}

### USER QUESTION:
{question}

### RESPONSE FORMAT:
- Provide a professional, clinical synthesis.
- Use a single coherent response (no forced headers).
- Ensure safety warnings and specific clinical exceptions are prominent.
"""
    return ChatPromptTemplate.from_template(template)


# ==============================
# GENERATE ANSWER
# ==============================

def generate_answer(question: str, context: str):
    llm = ChatGroq(
        temperature=0.0, # Zero temperature for clinical consistency
        model=GROQ_MODEL
    )
    prompt = get_prompt()
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })
    
    answer = response.content
    
    # Senior Engineering Post-Processing: 
    # Check for citations but allow for headers if the LLM reasonably included them
    # We maintain the citation pattern check to ensure every line is grounded.
    import re
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    
    # We only keep sentences that have a citation or are defining the context.
    # To be safe in clinical RAG, we stick to the cited sentences.
    filtered = []
    for s in sentences:
        if re.search(CITATION_PATTERN, s) or "Information not found" in s:
            filtered.append(s)
            
    # Remove duplicates while preserving order
    final = []
    seen = set()
    for s in filtered:
        if s not in seen:
            final.append(s)
            seen.add(s)
            
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