import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from fastembed import TextEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag.retriever")

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

class RAGRetriever:
    """
    Sub-millisecond semantic search against FAISS with strict confidence score thresholding.
    Prevents hallucination by refusing low-confidence matches.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        data_dir: str = "./data",
        similarity_threshold: float = 0.65
    ):
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.similarity_threshold = similarity_threshold
        self.index_file = self.data_dir / "faiss_index.bin"
        self.metadata_file = self.data_dir / "chunks.json"

        self._embedding_model: Optional[TextEmbedding] = None
        self._index: Optional[faiss.IndexFlatIP] = None
        self._chunks: List[Dict[str, Any]] = []

        self.reload()

    @property
    def embedding_model(self) -> TextEmbedding:
        if self._embedding_model is None:
            logger.info(f"Initializing FastEmbed runtime ({self.model_name})...")
            self._embedding_model = TextEmbedding(model_name=self.model_name)
        return self._embedding_model

    def reload(self):
        """Loads or reloads the FAISS index and chunk metadata from disk."""
        if self.index_file.exists() and self.metadata_file.exists():
            try:
                self._index = faiss.read_index(str(self.index_file))
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self._chunks = json.load(f)
                logger.info(f"RAGRetriever loaded {self._index.ntotal} vectors from {self.index_file}")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self._index = None
                self._chunks = []
        else:
            logger.warning(f"No FAISS index found at {self.index_file}. Ready to ingest documents.")
            self._index = None
            self._chunks = []

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Searches the knowledge base for chunks relevant to the user query.
        Returns:
            {
                "has_context": bool,
                "max_score": float,
                "chunks": List[Dict],
                "formatted_context": str
            }
        """
        cut_off = threshold if threshold is not None else self.similarity_threshold

        if not query or not query.strip():
            return {
                "has_context": False,
                "max_score": 0.0,
                "chunks": [],
                "formatted_context": "",
                "reason": "EMPTY_QUERY"
            }

        if self._index is None or self._index.ntotal == 0 or not self._chunks:
            return {
                "has_context": False,
                "max_score": 0.0,
                "chunks": [],
                "formatted_context": "",
                "reason": "EMPTY_INDEX"
            }

        # BGE models use asymmetric retrieval instruction
        search_query = query.strip()
        if "bge" in self.model_name.lower():
            search_query = f"Represent this sentence for searching relevant passages: {search_query}"

        # Embed query
        query_embedding_gen = self.embedding_model.embed([search_query])
        query_vec = np.array(list(query_embedding_gen), dtype=np.float32)
        faiss.normalize_L2(query_vec)

        # Search top-k
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, k)

        top_scores = scores[0]
        top_indices = indices[0]

        max_score = float(top_scores[0]) if len(top_scores) > 0 else 0.0

        # Enforce strict confidence threshold for hallucination control
        if max_score < cut_off:
            logger.info(f"Query '{query}' top similarity {max_score:.4f} below threshold {cut_off:.4f}. Rejecting context.")
            return {
                "has_context": False,
                "max_score": max_score,
                "chunks": [],
                "formatted_context": "",
                "reason": "LOW_CONFIDENCE"
            }

        # Filter chunks that clear the threshold
        valid_results = []
        context_blocks = []

        for score, idx in zip(top_scores, top_indices):
            if idx < 0 or idx >= len(self._chunks):
                continue
            if score >= cut_off:
                chunk_data = self._chunks[idx]
                valid_results.append({
                    "id": chunk_data["id"],
                    "source": chunk_data.get("source", "unknown"),
                    "score": float(score),
                    "text": chunk_data["text"]
                })
                context_blocks.append(f"[{chunk_data.get('source', 'Doc')}]: {chunk_data['text']}")

        formatted_context = "\n\n".join(context_blocks)
        logger.info(f"Retrieved {len(valid_results)} context chunks for '{query}' (Max score: {max_score:.4f})")

        return {
            "has_context": True,
            "max_score": max_score,
            "chunks": valid_results,
            "formatted_context": formatted_context,
            "reason": "OK"
        }
