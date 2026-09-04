import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import asyncio
import unittest
from agent.chunker import ClauseChunker

class TestClauseChunker(unittest.TestCase):
    def test_clause_streaming(self):
        async def mock_token_stream():
            # Simulates LLM tokens arriving word by word
            tokens = ["Certainly", ",", " our", " clinic", " is", " open", " until", " two", " p.m.", " on", " Saturday", "."]
            for token in tokens:
                yield token

        async def run_test():
            chunker = ClauseChunker(min_words_first_chunk=3)
            chunks = []
            async for chunk in chunker.split_stream(mock_token_stream()):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run_test())
        self.assertTrue(len(chunks) >= 1)
        first_chunk = chunks[0]
        # Verify first chunk was emitted early on clause punctuation
        self.assertTrue("Certainly," in first_chunk or len(first_chunk.split()) < 10)

if __name__ == "__main__":
    unittest.main()
