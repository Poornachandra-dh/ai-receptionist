import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Ensure root in path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
load_dotenv(dotenv_path=root_dir / ".env")

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import deepgram, silero, openai

from agent.prompt import RECEPTIONIST_SYSTEM_PROMPT
from rag.retriever import RAGRetriever
from api.db.supabase_client import async_log_turn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("agent.worker")

def prewarm_process(proc):
    """Pre-loads VAD model to eliminate first-call overhead."""
    logger.info("Pre-warming Silero VAD...")
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    logger.info(f"Connecting to room: {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # 1. Initialize RAG
    data_dir = os.getenv("RAG_DATA_DIR", "./data")
    threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.48"))
    retriever = RAGRetriever(data_dir=data_dir, similarity_threshold=threshold)

    # Function tool for real-time document grounding
    @llm.function_tool
    def lookup_knowledge_base(query: str) -> str:
        """Looks up clinic documents, operating hours, policies, insurance, and medical services."""
        logger.info(f"Grounding query against RAG: '{query}'")
        res = retriever.retrieve(query)
        if res.get("has_context"):
            return res["formatted_context"]
        return "NO_RECORDS_FOUND: The inquiry does not match any information in the clinic documents."

    # 2. Configure Agent with Receptionist persona & knowledge tool
    agent = Agent(
        instructions=RECEPTIONIST_SYSTEM_PROMPT.format(context="Use the lookup_knowledge_base tool to verify any clinic facts."),
        tools=[lookup_knowledge_base],
    )

    # 3. Configure Ultra-fast Deepgram STT (Nova-3 with 160ms endpointing)
    stt_engine = deepgram.STT(
        model=os.getenv("DEEPGRAM_STT_MODEL", "nova-3"),
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        interim_results=True,
        smart_format=True,
        endpointing_ms=160,
    )

    # 4. Configure Groq Cloud LLM (~100ms TTFT)
    llm_engine = openai.LLM(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=0.3,
    )

    # 5. Configure Streaming TTS (Deepgram Aura)
    tts_engine = deepgram.TTS(
        model=os.getenv("DEEPGRAM_TTS_MODEL", "aura-asteria-en"),
        api_key=os.getenv("DEEPGRAM_API_KEY")
    )

    vad = ctx.proc.userdata.get("vad") or silero.VAD.load()

    # 6. Configure AgentSession
    session = AgentSession(
        stt=stt_engine,
        llm=llm_engine,
        tts=tts_engine,
        vad=vad,
        turn_handling={
            "min_endpointing_delay": 0.16, # 160ms turn-taking
            "allow_interruptions": True,   # Instant barge-in
        },
    )

    # Start session in room
    await session.start(agent, room=ctx.room)

    # Initial greeting
    await asyncio.sleep(0.5)
    await session.say("Hello! Thanks for calling Apex Health Clinic. How can I help you today?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            agent_name="ai-receptionist",
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm_process
        )
    )
