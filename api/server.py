import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure root in path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
load_dotenv(dotenv_path=root_dir / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.token_route import router as token_router
from api.routes.documents_route import router as documents_router

app = FastAPI(
    title="AI Voice Receptionist API",
    description="Sub-500ms voice agent backend providing token generation, document ingestion, and telemetry.",
    version="1.0.0"
)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(token_router)
app.include_router(documents_router)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ai-voice-receptionist",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api.server:app", host="0.0.0.0", port=port, reload=True)
