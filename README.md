<div align="center">
  <h1>🩺 Clinical RAG System v2.0</h1>
  <p><i>An Enterprise-Grade, Hallucination-Resistant AI Medical Assistant</i></p>
</div>

---

## 📖 Overview
The **Clinical RAG System** is a production-ready, highly accurate Retrieval-Augmented Generation pipeline designed explicitly for the medical domain. It ingests thousands of clinical guidelines and protocols, providing healthcare professionals with precise, verified answers backed by interactive citations.

Unlike standard chatbots, this system leverages advanced RAG techniques—including semantic reranking and Natural Language Inference (NLI) grounding—to mathematically verify claims and prevent "medical hallucinations."

---

## ✨ Key Features
*   **Adaptive Dense Retrieval**: Multi-provider embedding support (OpenAI / Google Gemini) with built-in fallback logic for high availability.
*   **Zero-Shot Semantic Reranking**: Utilizes lightweight `FlashRank` to resort vector similarities, ensuring the most clinically relevant chunks always reach the LLM.
*   **NLI Grounding Verification**: A secondary LLM processing layer mathematically verifies generated claims against the original source texts to strictly enforce faithfulness.
*   **Interactive Evidence GUI**: A professional, responsive Vanilla JS frontend serving as a unified Clinical Dashboard. Features clickable `[Source X]` citations that open modals displaying the exact medical text.
*   **Conversational Memory**: Maintains persistent session states and implements intelligent "Query Contextualization" to support multi-turn clinical diagnostic conversations.
*   **Data Reproduction Strictness**: Engineered System Prompts that force the AI to extract and reproduce specific data/table values rather than lazily referencing them.

---

## 🛠️ Technology Stack
*   **Backend / API**: FastAPI (Python), Uvicorn
*   **RAG Engine & Orchestration**: LangChain, ChromaDB
*   **Inference & AI Models**: Meta Llama-3-70B (via Groq API)
*   **Embeddings**: OpenAI (`text-embedding-3-small`), Google Gemini Fallback
*   **Evaluation Framework**: Ragas (Faithfulness, Precision, Recall metrics)
*   **DevOps & Deployment**: Docker, AWS (EC2, ECR)

---

## 🗺️ Project Roadmap & Completed Milestones

### Phase 1: Core Data Ingestion & Indexing (✅ Completed)
- [x] Medical PDF extraction and semantic text chunking.
- [x] Vector Database initialization (`ChromaDB`).
- [x] Development of robust Embedding Functions (`OpenAIEmbeddingFunction` with `Gemini` fallback).

### Phase 2: Advanced RAG Pipeline (✅ Completed)
- [x] Integration of Cross-Encoder Semantic Rerankers (`FlashRank`).
- [x] Implementation of the NLI (Natural Language Inference) layer for hallucination checks.
- [x] Prompt Engineering for strict Data Reproduction from clinical tables.
- [x] Contextual Query Rewriting for seamless conversational memory.

### Phase 3: Application UI & Backend (✅ Completed)
- [x] Development of RESTful API endpoints using `FastAPI`.
- [x] Creation of a "Medical Grade", highly responsive Vanilla JS frontend.
- [x] Integration of the Source Identification Modal for real-time evidence viewing.

### Phase 4: DevOps & Scalable AWS Deployment (🚧 In Progress / Upcoming)
We have successfully containerized the application and achieved "Day 1" deployment on an AWS EC2 instance. Our professional deployment roadmap proceeds as follows:
1.  **Containerization**: [x] Package application via unified `Dockerfile`.
2.  **EC2 Provisioning**: [x] Launch Linux EC2 instance, install Docker, and configure Security Groups (Port 10000).
3.  **Persistent Storage**: Attach an AWS EBS (Elastic Block Store) volume and mount it via Docker (`-v /data/vector_store`) to ensure the ChromaDB guidelines survive container restarts.
4.  **Reverse Proxy & Networking**: Install `Nginx` on the EC2 instance to route traffic from Port 80 to Docker Port 10000.
5.  **Domain & Security**: Register a professional `.com` domain names via AWS Route 53 and provision wildcard SSL/TLS Certificates (HTTPS) using Let's Encrypt or AWS Certificate Manager.
6.  **CI/CD Automation**: Implement GitHub Actions to automatically rebuild the Docker image, push it to AWS Elastic Container Registry (ECR), and restart the EC2 container on every `git push`.

---

## ⚙️ Local Development Setup

1. **Clone & Environment**
   ```bash
   git clone <your-repo-link>
   cd Clinical_RAG_SYSTEM
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. **Install Top-Level Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Secrets**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_key
   OPENAI_API_KEY=your_openai_key
   GEMINI_API_KEY=your_gemini_key
   ```

4. **Launch Server**
   ```bash
   python server.py
   ```
   *Navigate to `http://localhost:8000` to view the UI.*

---
> *Architected for reliability, built for precision.*
