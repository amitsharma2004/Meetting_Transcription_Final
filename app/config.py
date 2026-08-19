"""Application configuration loaded from environment variables."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings for standalone voice transcription."""

    # App Info
    APP_NAME: str = "Voice Transcription & Speaker Recognition"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Database
    DATABASE_PATH: str = str(Path(__file__).parent.parent / "data" / "voice_transcription.db")

    # API Keys
    GOOGLE_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    SARVAM_API_KEY: str = ""

    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
        "*"
    ]

    # LLM Settings (for turn cleanup)
    LLM_PROVIDER: str = "groq"
    LLM_MODEL: str = "openai/gpt-oss-120b"
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GEMINI_MODEL: str = "gemini-1.5-flash"
    LLM_TIMEOUT: int = 6000

    # STT Settings (sarvam, openai, whisper)
    STT_PROVIDER: str = "sarvam"          # "sarvam", "openai", or "whisper"
    SARVAM_MODEL: str = "saarika:v2.5"    # saarika:v2.5 (Indic + Indian English SOTA)
    SARVAM_LANGUAGE: str = "unknown"      # "unknown" for auto-detect, "en-IN", "hi-IN"
    OPENAI_STT_MODEL: str = "whisper-1"   # "whisper-1" or "gpt-4o-transcribe"

    # Local Whisper Settings
    WHISPER_MODEL: str = "base"
    WHISPER_LANGUAGE: str = "en"
    WHISPER_DEVICE: str = "cpu"

    # ECAPA-TDNN Speaker Recognition Settings
    SPEAKER_SIMILARITY_THRESHOLD: float = 0.40  # Threshold calibrated via EER benchmark
    SPEAKER_SIMILARITY_MARGIN: float = 0.12     # Runner-up ambiguity protection margin

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
