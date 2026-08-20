"""Standalone Real-Time Voice Transcription & Speaker Recognition Application."""

from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import init_db
from app.routers import transcript, speakers, stt_lab


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    init_db()
    print("✓ Standalone database initialized.")
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Real-Time Multilingual Voice Transcription & Speaker Biometrics (Sarvam AI / OpenAI / Whisper + ECAPA-TDNN)",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(transcript.router, prefix="/api", tags=["Transcription"])
app.include_router(speakers.router, prefix="/api", tags=["Speakers"])
app.include_router(stt_lab.router, prefix="/api", tags=["STT Lab"])
app.include_router(stt_lab.router, prefix="", tags=["STT Lab"])


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "app": settings.APP_NAME, "version": settings.APP_VERSION}


@app.get("/", response_class=RedirectResponse)
async def root():
    """Redirect root to /voice-chat."""
    return RedirectResponse(url="/voice-chat")


@app.get("/voice-chat", response_class=FileResponse)
async def get_voice_chat_page():
    """Serve the standalone ChatGPT-style Voice Chat UI."""
    static_html = Path(__file__).parent.parent / "static" / "voice_chat.html"
    return FileResponse(static_html)


@app.get("/stt-lab", response_class=FileResponse)
async def get_stt_lab_page():
    """Serve the STT Lab Benchmarking & Comparison UI."""
    static_html = Path(__file__).parent.parent / "static" / "stt_lab.html"
    return FileResponse(static_html)
