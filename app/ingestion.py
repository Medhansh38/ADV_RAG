import io
import hashlib
import uuid
from datetime import datetime, timezone
from typing import List, Tuple
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.config import get_settings
from app.vectorstore import delete_document, client, get_embeddings

settings = get_settings()

def calculate_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def load_text_pages(content: bytes, filename: str) -> List[Tuple[str, int]]:
    lower = filename.lower()
    pages: List[Tuple[str, int]] = []
    if lower.endswith(".pdf"):
        reader = pypdf.PdfReader(io.BytesIO(content))
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((text, idx + 1))
    elif lower.endswith((".txt", ".md")):
        text = content.decode("utf-8", errors="replace")
        if text.strip():
            pages.append((text, 1))
    else:
        raise ValueError(f"Unsupported file type: {filename}")
    return pages

def build_chunks(pages: List[Tuple[str, int]], doc_id: str, filename: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks: List[Document] = []
    chunk_idx = 0
    now_iso = datetime.now(timezone.utc).isoformat()

    for text, page_num in pages:
        split_texts = splitter.split_text(text)
        for chunk_text in split_texts:
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "doc_id": doc_id,
                    "source": filename,
                    "page": page_num,
                    "chunk_index": chunk_idx,
                    "ingested_at": now_iso
                }
            )
            chunks.append(doc)
            chunk_idx += 1
    return chunks

def ingest_bytes(content: bytes, filename: str) -> Tuple[str, int, int]:
    doc_id = calculate_sha256(content)
    delete_document(doc_id)
    pages = load_text_pages(content, filename)
    if not pages:
        raise ValueError("No extractable text found in file")

    chunks = build_chunks(pages, doc_id, filename)
    if not chunks:
        raise ValueError("Could not create chunks from extracted content")

    embeddings_model = get_embeddings("retrieval_document")
    texts = [c.page_content for c in chunks]
    vectors = embeddings_model.embed_documents(texts)

    from qdrant_client.http import models as rest
    points = []
    for c, vec in zip(chunks, vectors):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}:{c.metadata['chunk_index']}"))
        points.append(
            rest.PointStruct(
                id=point_id,
                vector=vec,
                payload={"page_content": c.page_content, "metadata": c.metadata}
            )
        )

    client.upsert(collection_name=settings.COLLECTION_NAME, points=points)
    return doc_id, len(pages), len(chunks)