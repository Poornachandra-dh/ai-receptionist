import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger("api.db")

_supabase_client = None

def get_supabase():
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if url and key and "your-project" not in url:
        try:
            from supabase import create_client
            _supabase_client = create_client(url, key)
            logger.info("Connected to Supabase PostgreSQL database.")
        except Exception as e:
            logger.warning(f"Failed to initialize Supabase client ({e}). Falling back to local audit logging.")
            _supabase_client = None
    else:
        _supabase_client = None

    return _supabase_client

async def async_log_turn(
    room_name: str,
    user_text: str,
    has_context: bool,
    max_score: float,
    agent_text: Optional[str] = None
):
    """
    Non-blocking async telemetry logger.
    Logs to Supabase if configured; otherwise appends to local data/call_logs.jsonl.
    """
    record = {
        "room_name": room_name,
        "user_text": user_text,
        "has_context": has_context,
        "similarity_score": round(max_score, 4),
        "agent_text": agent_text,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    client = get_supabase()
    if client:
        try:
            client.table("call_turns").insert(record).execute()
            return
        except Exception as e:
            logger.error(f"Error inserting turn log to Supabase: {e}")

    # Fallback to local file logging
    try:
        log_dir = Path("./data")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "call_logs.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error(f"Failed to write local turn log: {e}")
