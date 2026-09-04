import os
import logging
from typing import Optional

logger = logging.getLogger("agent.tts")

def create_tts_engine(provider: Optional[str] = None):
    """
    Factory function for low-latency TTS.
    Defaults to Deepgram Aura (ultra-low latency ~110ms streaming via WebSocket).
    """
    provider = (provider or os.getenv("TTS_PROVIDER", "deepgram")).lower()

    if provider == "deepgram":
        try:
            from livekit.plugins import deepgram
            api_key = os.getenv("DEEPGRAM_API_KEY")
            if not api_key:
                logger.warning("DEEPGRAM_API_KEY is not set. Deepgram Aura TTS requires a valid key.")
            
            logger.info("Initializing Deepgram Aura low-latency streaming TTS (model: aura-asteria-en)...")
            return deepgram.TTS(
                model=os.getenv("DEEPGRAM_TTS_MODEL", "aura-asteria-en"),
                api_key=api_key
            )
        except ImportError:
            logger.warning("livekit-plugins-deepgram not installed, falling back.")

    # Fallback to OpenAI-compatible or other configured TTS
    try:
        from livekit.plugins import openai
        logger.info("Initializing OpenAI TTS fallback...")
        return openai.TTS(model="tts-1", voice="alloy")
    except Exception as e:
        logger.error(f"Failed to initialize TTS engine: {e}")
        return None
