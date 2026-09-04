import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from fastembed import TextEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag.ingest")

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

class DocumentIngester:
    """
    Extracts text from PDF, DOCX, TXT, and Markdown files,
    chunks text with configurable overlap, generates dense ONNX embeddings via FastEmbed,
    and builds an optimized FAISS IndexFlatIP (cosine similarity) index.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, data_dir: str = "./data"):
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.data_dir / "faiss_index.bin"
        self.metadata_file = self.data_dir / "chunks.json"
        self._embedding_model: Optional[TextEmbedding] = None

    @property
    def embedding_model(self) -> TextEmbedding:
        if self._embedding_model is None:
            logger.info(f"Loading FastEmbed model '{self.model_name}' on CPU ONNX Runtime...")
            self._embedding_model = TextEmbedding(model_name=self.model_name)
        return self._embedding_model

    def extract_text(self, file_path: str) -> str:
        """Extracts text content from supported file types."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found at {file_path}")

        suffix = path.suffix.lower()
        extracted_text = ""

        if suffix in [".txt", ".md"]:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                extracted_text = f.read()

        elif suffix == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(path))
                pages = []
                for idx, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        pages.append(page_text.strip())
                extracted_text = "\n\n".join(pages)
            except Exception as e:
                raise ValueError(f"Failed to extract text from PDF {path.name}: {e}")

        elif suffix == ".docx":
            try:
                import docx
                doc = docx.Document(str(path))
                extracted_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            except Exception as e:
                raise ValueError(f"Failed to extract text from DOCX {path.name}: {e}")

        else:
            raise ValueError(f"Unsupported file format '{suffix}'. Supported formats: .pdf, .docx, .txt, .md")

        cleaned = extracted_text.strip()
        if not cleaned:
            raise ValueError(f"Document {path.name} is empty or contains no readable text.")

        return cleaned

    def chunk_text(self, text: str, chunk_size_chars: int = 700, overlap_chars: int = 100) -> List[str]:
        """
        Splits text into cohesive semantic chunks.
        Respects Markdown headings and paragraph boundaries for maximum retrieval precision.
        """
        if not text or not text.strip():
            return []

        import re
        # Check if text contains markdown headers
        has_headers = bool(re.search(r"^#{1,4}\s+", text, re.MULTILINE))

        raw_sections = []
        if has_headers:
            # Split by markdown headers while keeping the header with the content
            pattern = r"(?=^#{1,4}\s+)"
            sections = re.split(pattern, text, flags=re.MULTILINE)
            raw_sections = [s.strip() for s in sections if s.strip()]
        else:
            raw_sections = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks: List[str] = []
        for section in raw_sections:
            if len(section) <= chunk_size_chars:
                chunks.append(section)
            else:
                # Sub-chunk long sections by paragraphs/sentences
                lines = [l.strip() for l in section.split("\n") if l.strip()]
                current_chunk = []
                current_len = 0
                for line in lines:
                    line_len = len(line)
                    if current_len + line_len > chunk_size_chars and current_chunk:
                        chunk_text = "\n".join(current_chunk)
                        chunks.append(chunk_text)
                        overlap = chunk_text[-overlap_chars:] if len(chunk_text) > overlap_chars else ""
                        current_chunk = [overlap, line] if overlap else [line]
                        current_len = len(overlap) + line_len
                    else:
                        current_chunk.append(line)
                        current_len += line_len + 1
                if current_chunk:
                    chunks.append("\n".join(current_chunk))

        return chunks

    def ingest_file(self, file_path: str, chunk_size_chars: int = 1500, overlap_chars: int = 200) -> Dict[str, Any]:
        """
        Extracts, chunks, embeds, and indexes a single file.
        Updates or creates the FAISS index and chunks metadata.
        """
        path = Path(file_path)
        logger.info(f"Ingesting file: {path.name}")
        raw_text = self.extract_text(file_path)
        chunks = self.chunk_text(raw_text, chunk_size_chars=chunk_size_chars, overlap_chars=overlap_chars)

        if not chunks:
            raise ValueError(f"No valid chunks could be created from {path.name}")

        return self.ingest_chunks(chunks, source_name=path.name)

    def ingest_chunks(self, chunks: List[str], source_name: str = "manual_entry") -> Dict[str, Any]:
        """
        Embeds a list of text chunks and saves them to the FAISS index and metadata store.
        """
        logger.info(f"Generating FastEmbed embeddings for {len(chunks)} chunks from '{source_name}'...")
        # Embed with FastEmbed generator
        embeddings_gen = self.embedding_model.embed(chunks)
        embeddings = np.array(list(embeddings_gen), dtype=np.float32)

        # Normalize vectors for Cosine Similarity
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]

        # Load existing index & metadata if present
        existing_chunks: List[Dict[str, Any]] = []
        if self.index_file.exists() and self.metadata_file.exists():
            try:
                index = faiss.read_index(str(self.index_file))
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    existing_chunks = json.load(f)
                logger.info(f"Loaded existing index with {index.ntotal} vectors.")
            except Exception as e:
                logger.warning(f"Could not load existing index ({e}), creating fresh index.")
                index = faiss.IndexFlatIP(dimension)
                existing_chunks = []
        else:
            index = faiss.IndexFlatIP(dimension)

        # Add new embeddings
        start_idx = len(existing_chunks)
        index.add(embeddings)

        for idx, chunk_text in enumerate(chunks):
            existing_chunks.append({
                "id": start_idx + idx,
                "source": source_name,
                "text": chunk_text
            })

        # Save to disk
        faiss.write_index(index, str(self.index_file))
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(existing_chunks, f, indent=2)

        logger.info(f"Successfully saved {index.ntotal} total vectors to {self.index_file}")
        return {
            "source": source_name,
            "new_chunks_added": len(chunks),
            "total_vectors": index.ntotal,
            "dimension": dimension
        }
