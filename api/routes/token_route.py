import os
import uuid
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("api.token")
router = APIRouter(prefix="/api/token", tags=["token"])

class TokenRequest(BaseModel):
    room_name: str = Field(default_factory=lambda: f"reception-{uuid.uuid4().hex[:8]}")
    identity: str = Field(default_factory=lambda: f"caller-{uuid.uuid4().hex[:6]}")
    name: str = "Web Caller"

class TokenResponse(BaseModel):
    token: str
    room_name: str
    identity: str
    server_url: str

@router.post("", response_model=TokenResponse)
async def generate_token(req: TokenRequest):
    """Generates a LiveKit JWT token for client WebRTC room connection."""
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    server_url = os.getenv("LIVEKIT_URL", "wss://your-project.livekit.cloud")

    if not api_key or not api_secret:
        raise HTTPException(
            status_code=500,
            detail="LIVEKIT_API_KEY or LIVEKIT_API_SECRET is not configured."
        )

    try:
        from livekit.api import AccessToken, VideoGrants
        token = (
            AccessToken(api_key, api_secret)
            .with_identity(req.identity)
            .with_name(req.name)
            .with_grants(VideoGrants(room_join=True, room=req.room_name))
            .to_jwt()
        )

        return TokenResponse(
            token=token,
            room_name=req.room_name,
            identity=req.identity,
            server_url=server_url
        )
    except Exception as e:
        logger.error(f"Failed to create LiveKit token: {e}")
        raise HTTPException(status_code=500, detail=str(e))
