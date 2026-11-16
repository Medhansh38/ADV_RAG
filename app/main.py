import uuid
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.schemas import QueryRequest, QueryResponse, IngestionResponse
from app.security import verify_api_key
from app.vectorstore import ensure_collection, list_documents, delete_document
from app.ingestion import ingest_bytes
from app.rag import run_rag_query
from fastapi.responses import FileResponse


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ensure_collection()
        logger.info("Vector collection ready.")
    except Exception as e:
        logger.error(f"Startup check failed: {e}")
    yield

app = FastAPI(
    title="Enterprise RAG System API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def logging_and_request_id_middleware(request: Request, call_next):
    req_id = str(uuid.uuid4())
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Request-ID"] = req_id
    logger.info(f"ID={req_id} Method={request.method} Path={request.url.path} Status={response.status_code} Duration={duration}ms")
    return response

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok"}

@app.post("/ingest", response_model=IngestionResponse, dependencies=[Depends(verify_api_key)], tags=["RAG"])
def ingest_file(file: UploadFile = File(...)):
    allowed_extensions = (".pdf", ".txt", ".md")
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file format. Supported: {allowed_extensions}"
        )

    content = file.file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum upload limit of {settings.MAX_UPLOAD_SIZE_MB}MB"
        )

    try:
        doc_id, pages, chunks = ingest_bytes(content, file.filename)
        return IngestionResponse(doc_id=doc_id, filename=file.filename, pages=pages, chunks=chunks)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as ex:
        logger.error(f"Ingestion failed: {ex}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Ingestion pipeline upstream failure")

@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)], tags=["RAG"])
def query_documents(request: QueryRequest):
    try:
        return run_rag_query(request)
    except Exception as ex:
        logger.error(f"Query generation failed: {ex}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Upstream inference failure")

@app.get("/documents", dependencies=[Depends(verify_api_key)], tags=["RAG"])
def get_documents():
    return list_documents()

@app.delete("/documents/{doc_id}", dependencies=[Depends(verify_api_key)], tags=["RAG"])
def remove_document(doc_id: str):
    delete_document(doc_id)
    return {"detail": f"Document {doc_id} deleted"}

@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse("app/static/index.html")