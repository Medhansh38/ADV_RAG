from typing import List, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from app.config import get_settings

settings = get_settings()

client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None
)

def get_embeddings(task_type: str = "retrieval_document") -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        task_type=task_type
    )

def ensure_collection() -> None:
    collections = [c.name for c in client.get_collections().collections]
    if settings.COLLECTION_NAME not in collections:
        probe_vector = get_embeddings("retrieval_document").embed_query("probe")
        dim = len(probe_vector)
        client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=rest.VectorParams(size=dim, distance=rest.Distance.COSINE)
        )
        client.create_payload_index(
            collection_name=settings.COLLECTION_NAME,
            field_name="metadata.doc_id",
            field_schema=rest.PayloadSchemaType.KEYWORD
        )

def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name=settings.COLLECTION_NAME,
        embedding=get_embeddings("retrieval_query")
    )

def delete_document(doc_id: str) -> None:
    ensure_collection()
    client.delete(
        collection_name=settings.COLLECTION_NAME,
        points_selector=rest.Filter(
            must=[
                rest.FieldCondition(
                    key="metadata.doc_id",
                    match=rest.MatchValue(value=doc_id)
                )
            ]
        )
    )
def list_documents() -> List[dict]:
    points, _ = client.scroll(
        collection_name=settings.COLLECTION_NAME,
        limit=10000,
        with_payload=True,
        with_vectors=False
    )
    docs_map = {}
    for pt in points:
        meta = pt.payload.get("metadata", {})
        doc_id = meta.get("doc_id")
        if doc_id:
            if doc_id not in docs_map:
                docs_map[doc_id] = {
                    "doc_id": doc_id,
                    "source": meta.get("source", "unknown"),
                    "chunks": 0,
                    "ingested_at": meta.get("ingested_at")
                }
            docs_map[doc_id]["chunks"] += 1
    return list(docs_map.values())