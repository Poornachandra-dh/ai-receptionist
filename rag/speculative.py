import asyncio
import logging
from typing import Optional, Dict, Any

from rag.retriever import RAGRetriever

logger = logging.getLogger("rag.speculative")

class SpeculativeRAGCache:
    """
    Speculative RAG Cache:
    Triggers concurrent vector searches on interim STT transcripts as the user speaks.
    By the time the user stops talking (VAD triggers), the candidate context is already
    in memory, reducing net retrieval latency to 0ms.
    """

    def __init__(self, retriever: RAGRetriever, min_words: int = 3):
        self.retriever = retriever
        self.min_words = min_words
        self._cached_context: Optional[Dict[str, Any]] = None
        self._last_query: str = ""
        self._pending_task: Optional[asyncio.Task] = None

    def on_interim_transcript(self, partial_text: str):
        """Called whenever an interim transcript arrives from Deepgram."""
        clean_text = partial_text.strip()
        words = clean_text.split()

        if len(words) < self.min_words:
            return

        if clean_text == self._last_query:
            return

        self._last_query = clean_text

        # Cancel any in-flight speculative task to avoid CPU waste
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()

        # Launch speculative background search
        try:
            loop = asyncio.get_running_loop()
            self._pending_task = loop.create_task(self._async_speculate(clean_text))
        except RuntimeError:
            pass

    async def _async_speculate(self, query: str):
        """Asynchronously runs retrieval in an executor without blocking event loop."""
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self.retriever.retrieve, query)
            if result.get("has_context"):
                self._cached_context = result
                logger.debug(f"Speculative cache updated for: '{query}'")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Speculative RAG error on '{query}': {e}")

    def get_final_context(self, final_text: str) -> Dict[str, Any]:
        """
        Retrieves context for final utterance.
        If cached speculative context is valid, returns it immediately (0ms).
        Otherwise falls back to synchronous retrieval.
        """
        clean_text = final_text.strip()

        # Check if cache matches or closely matches
        if self._cached_context and self._cached_context.get("has_context"):
            cached_query = self._last_query.lower()
            if cached_query in clean_text.lower() or clean_text.lower() in cached_query:
                logger.info(f"Speculative cache HIT for final text: '{clean_text}' (0ms retrieval)")
                return self._cached_context

        # Cache miss or query changed significantly -> direct fetch
        logger.info(f"Speculative cache MISS. Direct fetch for: '{clean_text}'")
        result = self.retriever.retrieve(clean_text)
        return result

    def reset(self):
        """Clears cache at the end of the turn."""
        self._cached_context = None
        self._last_query = ""
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()
            self._pending_task = None
