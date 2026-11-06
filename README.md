# Enterprise RAG System

A production-style Retrieval-Augmented Generation (RAG) backend engineered with **FastAPI**, **LangChain**, **Qdrant**, and **Google Gemini**. The system indexes unstructured enterprise documentation (PDF, Markdown, TXT) and serves citation-backed, hallucination-resistant answers via a secure REST API.

---

## Architecture Overview

```text
[INGESTION PIPELINE]
Uploaded File (PDF / TXT / MD)
       │
       ▼
SHA-256 Hashing ──► doc_id (Ensures Idempotency)
       │
       ▼
RecursiveCharacterTextSplitter (1000 chars, 150 overlap)
       │
       ▼
Gemini Embeddings (task_type: "retrieval_document")
       │
       ▼
Qdrant Vector DB (HNSW Cosine Index + Payload Index on metadata.doc_id)

──────────────────────────────────────────────────────────────────────────

[QUERY PIPELINE]
User Question
       │
       ▼
Gemini Query Embedding (task_type: "retrieval_query")
       │
       ▼
Qdrant Vector Search (Cosine Similarity, Top-K = 4)
       │
       ▼
Score Threshold Filter (score >= 0.40)
       ├──────────────────────────────────────────┐
       │ (No chunks pass threshold)               │ (Chunks pass)
       ▼                                          ▼
Fallback: "I don't know..."                Construct Grounded Prompt [1], [2]
(LLM Bypassed: 0 Tokens, <40ms)                   │
                                                  ▼
                                           Gemini 2.5 Flash (temp = 0.1)
                                                  │
                                                  ▼
                                           Answer with Inline Citations