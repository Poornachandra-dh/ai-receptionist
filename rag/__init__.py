"""
Ultra-fast RAG package using FastEmbed (ONNX) and FAISS.
Designed for sub-10ms vector retrieval and speculative pre-fetching.
"""

from rag.ingest import DocumentIngester
from rag.retriever import RAGRetriever
from rag.speculative import SpeculativeRAGCache

__all__ = ["DocumentIngester", "RAGRetriever", "SpeculativeRAGCache"]
