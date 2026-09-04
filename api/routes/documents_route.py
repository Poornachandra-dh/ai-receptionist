import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any

from fastapi import APIRouter, UploadFile, File, HTTPException
from rag.ingest import DocumentIngester
from rag.retriever import RAGRetriever

logger = logging.getLogger("api.documents")
router = APIRouter(prefix="/api/documents", tags=["documents"])

UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Uploads a document (PDF, DOCX, TXT, MD) and indexes it into FAISS."""
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    allowed_exts = {".pdf", ".docx", ".txt", ".md"}
    ext = Path(filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed_exts)}"
        )

    dest_path = UPLOAD_DIR / filename
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Ingest file into FAISS
        data_dir = os.getenv("RAG_DATA_DIR", "./data")
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        ingester = DocumentIngester(model_name=model_name, data_dir=data_dir)

        result = ingester.ingest_file(str(dest_path))
        return {
            "status": "success",
            "filename": filename,
            "details": result
        }
    except Exception as e:
        logger.error(f"Ingestion failed for {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_index_status() -> Dict[str, Any]:
    """Returns the current status of the indexed knowledge base."""
    data_dir = os.getenv("RAG_DATA_DIR", "./data")
    retriever = RAGRetriever(data_dir=data_dir)
    return {
        "total_vectors": retriever._index.ntotal if retriever._index else 0,
        "total_chunks": len(retriever._chunks),
        "data_directory": str(Path(data_dir).resolve())
    }
