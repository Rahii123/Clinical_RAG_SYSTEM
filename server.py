import os
import re
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Load original RAG logic
from src.rag_pipeline import (
    load_vector_db, 
    retrieve_documents, 
    build_context, 
    generate_answer, 
    verify_citations,
    contextualize_query,
    CITATION_PATTERN
)

load_dotenv()

app = FastAPI(title="Clinical RAG API")

# Enable CORS for local UI development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Frontend Files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse('static/index.html')

# Initialize DB once
vectordb = load_vector_db()

# In-memory session store
chat_histories = {}

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default_user"

class SourceMetadata(BaseModel):
    id: int
    title: str
    section: str
    year: int
    content: str # Added to support interactive citation previews

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceMetadata]
    grounding: str
    transparency_log: List[str]

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    if not vectordb:
        raise HTTPException(status_code=500, detail="Vector Database not initialized.")

    query = request.query
    session_id = request.session_id

    # 1. Access Session History
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    history = chat_histories[session_id]

    # 2. Contextualize (Rewrite) Query based on memory
    standalone_query = contextualize_query(query, history)
    
    # 3. Retrieval (using the standalone query)
    docs = retrieve_documents(standalone_query, vectordb)
    if not docs:
        return QueryResponse(
            answer="No relevant documents found for this query.",
            sources=[],
            grounding="UNVERIFIED",
            transparency_log=["Zero results matching threshold."]
        )

    # 4. Context Building
    context = build_context(docs)
    
    # 5. Answer Generation
    raw_answer = generate_answer(query, context, docs)
    clean_answer = raw_answer.split("---")[0].strip()

    # 6. Verify Grounding
    verification = verify_citations(clean_answer, context)
    grounding_status = "VERIFIED" if "VERIFIED" in verification.upper() else "WARNING"

    # 7. Update History
    chat_histories[session_id].append({"role": "user", "content": query})
    chat_histories[session_id].append({"role": "assistant", "content": clean_answer})

    # 8. Format Sources for UI
    ui_sources = []
    for i, doc in enumerate(docs, 1):
        ui_sources.append(SourceMetadata(
            id=i,
            title=doc.metadata.get("guideline_name") or "Unknown Guideline",
            section=doc.metadata.get("section_header") or "General",
            year=int(doc.metadata.get("year", 0)),
            content=doc.page_content # Include the actual text for previews
        ))

    return QueryResponse(
        answer=clean_answer,
        sources=ui_sources,
        grounding=grounding_status,
        transparency_log=[f"Rewritten: {standalone_query}", f"Retrieved {len(docs)} segments"]
    )

if __name__ == "__main__":
    import uvicorn
    # Render provides the PORT via environment variable
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
