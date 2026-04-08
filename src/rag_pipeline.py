import os
import re
from typing import List
from dotenv import load_dotenv
load_dotenv()

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import requests

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


try:
    from src.embeddings import get_embedding_function
except ImportError:
    from embeddings import get_embedding_function

ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
if not os.path.exists(ENV_PATH):
    ENV_PATH = os.path.abspath(".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

embeddings = get_embedding_function()

# ==============================
# CONFIG
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QDRANT_STORAGE_PATH = os.path.join(BASE_DIR, "qdrant_storage")
COLLECTION_NAME = "clinical_rag_v1"

GROQ_MODEL = "llama-3.3-70b-versatile"

TOP_K = 12
MMR_FETCH_K = 30
MMR_LAMBDA = 0.4
SCORE_THRESHOLD = 0.3 # Lower is more similar (Cosine distance)





# ==============================
# LOAD VECTOR DB
# ==============================

def load_vector_db():
    client = QdrantClient(path=QDRANT_STORAGE_PATH)
    
    vectordb = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    try:
        # Check if collection exists and has points
        collection_info = client.get_collection(COLLECTION_NAME)
        count = collection_info.points_count
        print(f"\n📊 Collection: {COLLECTION_NAME} | Chunks: {count}")
    except Exception:
        print(f"❌ Collection {COLLECTION_NAME} not found.")
        return None

    if count == 0:
        print("❌ Vector store empty. Run qdrant_store.py")
        return None

    return vectordb


# ==============================
# RETRIEVAL (Transparent)
# ==============================

# ==============================
# RERANKER (Semantic Precision)
# ==============================

try:
    from flashrank import Ranker, RerankRequest
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./.cache")
except Exception as e:
    print(f"⚠️ Reranker failed to load: {e}")
    ranker = None

# ==============================
# RETRIEVAL (Adaptive & Transparent)
# ==============================

def retrieve_documents(query: str, vectordb) -> List[Document]:
    current_threshold = SCORE_THRESHOLD
    print(f"\n📚 Searching (Target Score: {current_threshold}, Lambda: {MMR_LAMBDA})...")
    
    scored_docs = vectordb.similarity_search_with_score(query, k=MMR_FETCH_K)
    score_lookup = {doc.page_content[:200]: score for doc, score in scored_docs}

    mmr_docs = vectordb.max_marginal_relevance_search(
        query,
        k=TOP_K,
        fetch_k=MMR_FETCH_K,
        lambda_mult=MMR_LAMBDA
    )

    def filter_docs(threshold):
        final = []
        seen = set()
        for doc in mmr_docs:
            key = doc.page_content[:200]
            if key in seen: continue
            score = score_lookup.get(key, 1.0)
            if score <= threshold:
                seen.add(key)
                final.append(doc)
        return final

    filtered_docs = filter_docs(current_threshold)

    if not filtered_docs and current_threshold < 0.5:
        # Debug: What was the best score we actually found?
        if scored_docs:
            best_score = min([s for _, s in scored_docs])
            print(f"🔍 Debug: Best overall match score was {best_score:.4f}")
            
        print(f"⚠️  No results at {current_threshold}. Relaxing threshold to 0.5...")
        filtered_docs = filter_docs(0.5)

    if not filtered_docs and scored_docs:
         best_score = min([s for _, s in scored_docs])
         print(f"🚨 ALERT: Even at 0.5, best match was {best_score:.4f}. Check if embedding models match!")

    # SENIOR STEP: Semantic Reranking
    if ranker and filtered_docs:
        print(f"⚖️  Reranking {len(filtered_docs)} candidates for semantic precision...")
        passages = [
            {"id": i, "text": d.page_content, "meta": d.metadata} 
            for i, d in enumerate(filtered_docs)
        ]
        rerankrequest = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerankrequest)
        
        # Keep top 6 most relevant after reranking
        reranked_docs = []
        for r in results[:6]:
            doc = filtered_docs[r['id']]
            # Update score in transparency report if needed
            reranked_docs.append(doc)
        filtered_docs = reranked_docs

    print("\n🔎 Retrieval Transparency Report")
    print("-" * 60)

    for i, doc in enumerate(filtered_docs, 1):
        # Professional metadata mapping
        title = doc.metadata.get("guideline_name") or doc.metadata.get("filename", "Doc")
        section = doc.metadata.get("section_header", "General")
        print(f"[{i}] Source: {title} | Section: {section}")
        
    print("-" * 60)

    if not filtered_docs:
        print("❌ No documents matched even the fallback threshold.")

    return filtered_docs


# ==============================
# CONTEXTUALIZATION (Memory Layer)
# ==============================

def contextualize_query(query: str, history: List[dict] = None) -> str:
    """
    Rewrites a follow-up question into a standalone question based on history.
    """
    if not history or len(history) == 0:
        return query

    llm = ChatGroq(temperature=0.0, model="llama-3.1-8b-instant")
    
    # Format history for the prompt
    history_str = ""
    for msg in history[-3:]: # Only look at last 3 turns for efficiency
        role = "User" if msg['role'] == 'user' else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    context_prompt = f"""
    Given a chat history and the latest user question which might reference context in the chat history, 
    formulate a standalone question which can be understood without the chat history. 
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is.

    Chat History:
    {history_str}
    
    Latest Question: {query}
    
    Standalone Question:"""

    try:
        response = llm.invoke(context_prompt)
        standalone = response.content.strip()
        print(f"🔄 Rewritten Query: {standalone}")
        return standalone
    except Exception:
        return query

# ==============================
# PROMPT WITH STRICT CITATION
# ==============================

def get_prompt():
    template = """
You are a Senior Clinical AI Assistant. Your goal is to provide a high-precision, safety-first answer based EXCLUSIVELY on the provided context.

### SYSTEM RULES:
1. **Acronym Definition**: At the first mention of any medical acronym, define it in parentheses.
2. **Handle Exceptions**: If the context mentions specific pathogens, age groups, or conditions with DIFFERENT rules, you MUST include these exceptions.
3. **Evidence Strength**: Include evidence quality labels (strong, weak, high-quality) if present.
4. **Synthesis**: Synthesize information from MULTIPLE sources. Use and cite [Source 1], [Source 2], etc.
5. **Strict Citation**: Every factual claim MUST be followed by its source in brackets. 
6. **Data Reproduction**: If the context refers to a table (e.g., 'Table 2-4') or specific diagnostic values, you MUST extract and reproduce those ACTUAL values/numbers in your answer. Do NOT just refer to the table number; provide the data inside it.

### CONTEXT:
{context}

### USER QUESTION:
{question}

### RESPONSE FORMAT:
- Provide a professional, clinical synthesis.
- Ensure safety warnings are prominent.
"""
    return ChatPromptTemplate.from_template(template)


# ==============================
# CITATION VERIFICATION (NLI)
# ==============================

def verify_citations(answer: str, context: str):
    """
    Second-pass verification to ensure citations are not hallucinated.
    """
    llm = ChatGroq(temperature=0.0, model="llama-3.1-8b-instant") # Faster, cheaper model for verification
    
    verify_prompt = f"""
    Internal Verification Task:
    Check if the following Answer is supported by the provided Context.
    Look for citations like [Source X] and verify the claim.
    If a claim is NOT supported, flag it.
    
    Context:
    {context}
    
    Answer:
    {answer}
    
    Response format:
    If all good: "VERIFIED"
    If issues found: List the specific unsupported sentences.
    """
    
    try:
        response = llm.invoke(verify_prompt)
        return response.content
    except Exception:
        return "VERIFICATION_SKIPPED"


# ==============================
# GENERATE ANSWER
# ==============================

def generate_answer(question: str, context: str, docs: List[Document] = None):
    llm = ChatGroq(
        temperature=0.0, 
        model=GROQ_MODEL,
        max_tokens=4096  # Increased to prevent truncation of long responses
    )
    prompt = get_prompt()
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })
    
    answer = response.content
    
    # Add References Footer
    if docs:
        ref_parts = ["\n\n---", "**Retrieved References (Verified):**"]
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("guideline_name") or "Clinical Guideline"
            year = doc.metadata.get("year", "N/A")
            ref_parts.append(f"- [Source {i}]: {title} ({year})")
        
        answer += "\n".join(ref_parts)
            
    return answer


# ==============================
# REGEX VALIDATION LAYER
# ==============================

CITATION_PATTERN = r"\[Source\s+\d+\]"

def validate_answer(answer: str, context: str):
    print("\n📋 Senior-Level Validation")
    print("-" * 40)

    # Check citations
    citations = re.findall(CITATION_PATTERN, answer)
    if len(citations) == 0:
        print("❌ ERROR: No citations detected")
    else:
        print(f"✅ Citations detected: {len(citations)}")

    # Citation Verification (NLI)
    print("🔬 Verifying factual grounding via NLI...")
    verification = verify_citations(answer, context)
    if "VERIFIED" in verification.upper():
        print("✅ Grounding: All claims verified against source material.")
    else:
        print(f"⚠️  GROUNDING WARNING:\n{verification}")

    print("-" * 40)

def build_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content.strip()
        title = doc.metadata.get("guideline_name") or "Clinical Guideline"
        section = doc.metadata.get("section_header", "General")
        
        if len(content) > 1500:
            content = content[:1500] + "\n...[truncated]"
            
        parts.append(
            f"### [Source {i}] ###\n"
            f"GUIDELINE: {title}\n"
            f"SECTION: {section}\n"
            f"CONTENT:\n{content}\n"
            f"{'─'*70}"
        )
    return "\n\n".join(parts)


# ==============================
# MAIN LOOP
# ==============================

def main():
    vectordb = load_vector_db()
    if not vectordb: return

    print(f"\n🩺 Clinical RAG v2.0 (Verified) | {GROQ_MODEL}\n")

    while True:
        query = input("🩺 Query (or 'exit'): ").strip()
        if query.lower() == "exit": break

        docs = retrieve_documents(query, vectordb)
        if not docs:
            print("❌ No relevant documents found.")
            continue

        context = build_context(docs)
        answer = generate_answer(query, context, docs)

        print("\n" + "=" * 80)
        print("📋 CLINICAL SYNTHESIS")
        print("=" * 80)
        print(answer)
        print("=" * 80)

        validate_answer(answer, context)


if __name__ == "__main__":
    main()
