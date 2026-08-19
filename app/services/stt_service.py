"""STT service — audio file → faster-whisper → list[TranscriptSegment].

Uses faster-whisper (CTranslate2-based) instead of openai-whisper:
  - Pre-built wheels — no C++ build step needed
  - 4-8x faster on CPU with same accuracy
  - Same model names: tiny, base, small, medium, large-v2, large-v3

Speaker identity is NOT handled here — that happens in speaker_service.py (Step 3).

Usage:
    segments = await transcribe_audio(audio_bytes, filename="recording.wav")
    # [TranscriptSegment(text="Hello", start=0.0, end=2.4, speaker_id=None), ...]
"""

import asyncio
import tempfile
import os
from dataclasses import dataclass
from typing import Optional
import httpx

from app.config import settings

# Whisper model is loaded lazily — first call loads into memory, reused after.
_whisper_model = None


def _get_model():
    """Load faster-whisper model once and cache it."""
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            print(f"[stt_service] Loading faster-whisper model '{settings.WHISPER_MODEL}'...")
            _whisper_model = WhisperModel(
                settings.WHISPER_MODEL,
                device=settings.WHISPER_DEVICE,
                compute_type="int8",    # int8 is fast + low memory on CPU
            )
            print(f"[stt_service] faster-whisper model '{settings.WHISPER_MODEL}' loaded.")
        except ImportError:
            raise RuntimeError(
                "faster-whisper is not installed. "
                "Run: pip install faster-whisper"
            )
    return _whisper_model


@dataclass
class TranscriptSegment:
    """A single timed chunk of transcribed speech.

    speaker_id is None until speaker_service resolves it (Step 3).
    speaker_name and role are populated by enrich_segments() in speaker_service.
    """

    text: str
    start: float                         # seconds from audio start
    end: float                           # seconds from audio start
    speaker_id: Optional[str] = None    # e.g. "spk_mgr_meet001"
    speaker_name: Optional[str] = None  # e.g. "Arjun Mehta"  — set by enrich_segments()
    role: Optional[str] = None          # "manager" | "reportee" — set by enrich_segments()

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
            "role": self.role,
        }


def _run_whisper(audio_path: str) -> list[TranscriptSegment]:
    """
    Run faster-whisper synchronously on a file path.

    Returns one TranscriptSegment per segment (typically a sentence
    or natural pause boundary).
    """
    model = _get_model()

    transcribe_kwargs = {
        "beam_size": 5,
        "word_timestamps": False,
        "condition_on_previous_text": False,   # Prevent hallucination loops across chunks
        "vad_filter": True,                    # Use built-in Silero VAD for speech detection
        "vad_parameters": {
            "min_silence_duration_ms": 500,    # Silence interval to split on
            "speech_pad_ms": 200,              # Speech padding to preserve initial/final phonemes
        },
        "no_speech_threshold": 0.6,
        "compression_ratio_threshold": 2.4,
    }
    language = settings.WHISPER_LANGUAGE
    if language and language.lower() != "auto":
        transcribe_kwargs["language"] = language

    # faster-whisper returns a generator of segments + info
    segments_gen, info = model.transcribe(audio_path, **transcribe_kwargs)

    segments: list[TranscriptSegment] = []
    seen_texts = set()

    for seg in segments_gen:
        text = seg.text.strip()
        if not text:
            continue

        # Confidence filtering: ignore high no-speech probability or very low logprob
        no_speech_prob = getattr(seg, "no_speech_prob", 0.0)
        avg_logprob = getattr(seg, "avg_logprob", 0.0)
        if no_speech_prob > 0.6 or avg_logprob < -1.5:
            continue

        # Filter repeated hallucinated loops within the same chunk
        cleaned = text.lower().strip(". ,!?;:")
        if cleaned in seen_texts:
            continue
        seen_texts.add(cleaned)

        segments.append(
            TranscriptSegment(
                text=text,
                start=float(seg.start),
                end=float(seg.end),
            )
        )

    return segments


async def _run_sarvam_stt(audio_bytes: bytes, filename: str = "chunk.wav") -> list[TranscriptSegment]:
    """
    Transcribes audio using Sarvam AI Speech-to-Text API (Saaras v2).
    Optimized for Indian accents, Hinglish, Hindi, and regional languages.
    """
    api_key = settings.SARVAM_API_KEY
    if not api_key:
        raise ValueError("SARVAM_API_KEY is not configured in settings or .env")

    url = "https://api.sarvam.ai/speech-to-text"
    headers = {
        "api-subscription-key": api_key
    }

    ext = os.path.splitext(filename)[-1].lower() or ".wav"
    mime_type = "audio/webm" if "webm" in ext else "audio/wav"

    files = {
        "file": (f"audio{ext}", audio_bytes, mime_type)
    }
    data = {
        "model": getattr(settings, "SARVAM_MODEL", "saaras:v2"),
        "language_code": getattr(settings, "SARVAM_LANGUAGE", "unknown"),
        "with_timestamps": "true"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, headers=headers, files=files, data=data)
        if response.status_code != 200:
            raise RuntimeError(f"Sarvam API error ({response.status_code}): {response.text}")

        res_json = response.json()
        transcript = res_json.get("transcript", "").strip()
        if not transcript:
            return []

        # Extract timestamps from Sarvam schema
        timestamps = res_json.get("timestamps", {})
        start = 0.0
        end = 4.0
        if isinstance(timestamps, dict):
            starts = timestamps.get("start_time_seconds", [])
            ends = timestamps.get("end_time_seconds", [])
            if starts and isinstance(starts, list):
                start = float(starts[0])
            if ends and isinstance(ends, list):
                end = float(ends[-1])

        return [
            TranscriptSegment(
                text=transcript,
                start=round(start, 2),
                end=round(end, 2)
            )
        ]


async def _run_openai_stt(audio_bytes: bytes, filename: str = "chunk.wav") -> list[TranscriptSegment]:
    """
    Transcribes audio using OpenAI Audio API (gpt-live-transcribe / whisper-1) with verbose timestamps.
    """
    api_key = getattr(settings, "OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured in settings or .env")

    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    ext = os.path.splitext(filename)[-1].lower() or ".wav"
    mime_type = "audio/webm" if "webm" in ext else "audio/wav"

    files = {
        "file": (f"audio{ext}", audio_bytes, mime_type)
    }
    model_name = getattr(settings, "OPENAI_STT_MODEL", "whisper-1")
    data = {
        "model": model_name,
        "response_format": "verbose_json"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, headers=headers, files=files, data=data)
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI STT API error ({response.status_code}): {response.text}")

        res_json = response.json()
        raw_text = res_json.get("text", "").strip()
        if not raw_text:
            return []

        segments_data = res_json.get("segments", [])
        if segments_data and isinstance(segments_data, list):
            segments: list[TranscriptSegment] = []
            for seg in segments_data:
                t = seg.get("text", "").strip()
                if t:
                    segments.append(
                        TranscriptSegment(
                            text=t,
                            start=round(float(seg.get("start", 0.0)), 2),
                            end=round(float(seg.get("end", 0.0)), 2),
                        )
                    )
            if segments:
                return segments

        # Fallback to full duration if no detailed segments
        duration = float(res_json.get("duration", 4.0))
        return [
            TranscriptSegment(
                text=raw_text,
                start=0.0,
                end=round(duration, 2)
            )
        ]


async def transcribe_audio(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    provider: Optional[str] = None,
) -> list[TranscriptSegment]:
    """
    Unified STT router:
    - If provider == 'sarvam' (or default) and SARVAM_API_KEY is set -> Calls Sarvam AI API
    - If provider in {'openai', 'gpt-live-transcribe', 'whisper'} -> Calls OpenAI gpt-live-transcribe API
    - Fallback -> Gracefully falls back to local faster-whisper on CPU
    """
    selected_provider = (provider or getattr(settings, "STT_PROVIDER", "sarvam")).lower()

    # 1. Sarvam AI (Saarika v2.5)
    if selected_provider == "sarvam" and getattr(settings, "SARVAM_API_KEY", ""):
        try:
            return await _run_sarvam_stt(audio_bytes, filename)
        except Exception as e:
            print(f"[stt_service] Sarvam API error: {e} -> Falling back to faster-whisper.")

    # 2. OpenAI Cloud STT (gpt-live-transcribe / whisper-1)
    if selected_provider in {"openai", "gpt-live-transcribe", "whisper", "whisper_api"}:
        if getattr(settings, "OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", ""):
            try:
                return await _run_openai_stt(audio_bytes, filename)
            except Exception as e:
                print(f"[stt_service] OpenAI STT error: {e} -> Falling back to faster-whisper.")

    # 3. Local faster-whisper on CPU
    ext = os.path.splitext(filename)[-1].lower() or ".wav"
    if ext not in {".wav", ".mp3", ".mp4", ".m4a", ".webm", ".ogg", ".flac"}:
        ext = ".wav"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()
        segments = await loop.run_in_executor(None, _run_whisper, tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return segments
