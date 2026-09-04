import re
from typing import AsyncGenerator, List

CLAUSE_PUNCTUATION = re.compile(r"([,;:—\.\?!]+)")

class ClauseChunker:
    """
    Sub-sentence clause streaming tokenizer.
    Splits LLM token stream into early chunks (on commas, pauses, or first 4-5 words)
    so TTS audio generation begins within ~90ms without waiting for a full sentence period.
    """

    def __init__(self, min_words_first_chunk: int = 4, max_words_chunk: int = 12):
        self.min_words_first_chunk = min_words_first_chunk
        self.max_words_chunk = max_words_chunk

    async def split_stream(self, token_generator: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        """
        Yields synthesized speech chunks as soon as a clause boundary or word threshold is reached.
        """
        buffer = ""
        is_first_chunk = True

        async for token in token_generator:
            buffer += token
            words = buffer.strip().split()
            word_count = len(words)

            # Check for clause punctuation
            match = CLAUSE_PUNCTUATION.search(buffer)

            if match:
                # If there's punctuation and enough words, emit immediately
                punct_pos = match.end()
                chunk = buffer[:punct_pos].strip()
                remaining = buffer[punct_pos:]

                # For the very first chunk, even 3-4 words is enough to start audio playing
                threshold = self.min_words_first_chunk if is_first_chunk else 3
                if len(chunk.split()) >= threshold:
                    yield chunk
                    buffer = remaining
                    is_first_chunk = False
            elif is_first_chunk and word_count >= self.min_words_first_chunk:
                # First chunk reached word count without punctuation -> emit early to break silence
                yield buffer.strip()
                buffer = ""
                is_first_chunk = False
            elif word_count >= self.max_words_chunk:
                # Max words reached without punctuation -> emit chunk
                yield buffer.strip()
                buffer = ""

        # Flush any remaining text in buffer
        if buffer.strip():
            yield buffer.strip()
