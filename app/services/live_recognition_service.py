"""Live Recognition Service — Unified pipeline combining faster-whisper STT + ECAPA-TDNN Speaker ID.

Implements Step 4C & 4D:
  - Audio chunk -> faster-whisper (timed text segments with Silero VAD)
  - Audio slice -> ECAPA-TDNN (192-d embedding)
  - Embedding -> RAM meeting cache (instant dot product)
  - Decision Logic -> Threshold (0.50) + Margin (0.15) for Unknown/Uncertain protection
  - Merged output -> Enriched TranscriptSegments with speaker names, roles, scores, and status.
"""

from __future__ import annotations

import os
import io
import tempfile
import asyncio
import numpy as np
import av
from typing import List, Optional, Tuple, Dict, Any

from app.services.stt_service import transcribe_audio, TranscriptSegment
from app.services.speaker_profile_service import (
    extract_normalized_embedding,
    match_live_embedding_in_cache,
    active_meetings
)


def decode_audio_to_pcm16k(audio_bytes_or_path) -> Tuple[np.ndarray, float]:
    """
    Universally decodes ANY audio format (WebM, Opus, OGG, MP4, WAV) to 16kHz mono float32 array.
    Returns (audio_array, duration_seconds).
    """
    if isinstance(audio_bytes_or_path, bytes):
        container = av.open(io.BytesIO(audio_bytes_or_path))
    else:
        container = av.open(audio_bytes_or_path)

    resampler = av.AudioResampler(format='fltp', layout='mono', rate=16000)
    frames = []
    for frame in container.decode(audio=0):
        for resampled in resampler.resample(frame):
            frames.append(resampled.to_ndarray())
    container.close()

    if not frames:
        return np.array([], dtype=np.float32), 0.0

    audio_data = np.concatenate([f.squeeze() for f in frames]).astype(np.float32)
    duration = len(audio_data) / 16000.0
    return audio_data, duration


def extract_audio_slice(audio_data: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    """Extracts a slice from audio array by start/end timestamps."""
    start_idx = max(0, int(start_sec * sr))
    end_idx = min(len(audio_data), int(end_sec * sr))

    if end_idx <= start_idx:
        return audio_data

    slice_data = audio_data[start_idx:end_idx]
    # Ensure minimum 0.5s duration for ECAPA stability
    if len(slice_data) < int(0.5 * sr):
        pad_len = int(0.5 * sr) - len(slice_data)
        slice_data = np.pad(slice_data, (0, pad_len), mode="constant")

    return slice_data


async def transcribe_and_identify_speakers(
    audio_bytes: bytes,
    meeting_id: str,
    filename: str = "chunk.wav",
    threshold: float = 0.40,
    margin: float = 0.12,
    stt_provider: Optional[str] = "sarvam"
) -> List[dict]:
    """
    Step 4 Core Pipeline:
    Concurrently transcribes audio via STT Provider (Sarvam / Whisper) and recognizes speakers via ECAPA RAM Cache.
    
    Returns list of dicts:
      [
        {
          "start": 0.0,
          "end": 3.8,
          "text": "Let's review the deployment.",
          "speaker_id": "spk_001",
          "speaker_name": "Amit",
          "role": "manager",
          "speaker_score": 0.8913,
          "status": "MATCHED",
          "margin": 0.8617,
          "provider": "sarvam"
        }
      ]
    """
    # 1. Save audio to temporary file for both STT and SoundFile
    ext = os.path.splitext(filename)[-1].lower() or ".wav"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # 2. Run STT with chosen provider
        segments: List[TranscriptSegment] = await transcribe_audio(
            audio_bytes,
            filename=filename,
            provider=stt_provider
        )
        if not segments:
            return []

        # 3. If meeting_id is not in RAM cache, auto-load from database
        if meeting_id not in active_meetings:
            try:
                from app.services.speaker_profile_service import load_meeting_cache
                load_meeting_cache(meeting_id)
            except Exception:
                pass

        if meeting_id not in active_meetings:
            return [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "speaker_id": None,
                    "speaker_name": "Unknown",
                    "role": "guest",
                    "speaker_score": 0.0,
                    "status": "UNCACHED_MEETING",
                    "provider": stt_provider or "sarvam"
                }
                for seg in segments
            ]

        # 4. Load full audio waveform into 16kHz mono PCM for ECAPA segment slicing
        audio_data, duration = decode_audio_to_pcm16k(audio_bytes)
        sr = 16000

        enriched_results = []
        for seg in segments:
            # Slicing audio corresponding to segment timestamp
            seg_slice = extract_audio_slice(audio_data, sr, seg.start, seg.end)

            # Extract ECAPA embedding for this slice
            emb = extract_normalized_embedding(seg_slice)

            # Match against meeting RAM cache
            decision = match_live_embedding_in_cache(
                meeting_id=meeting_id,
                live_embedding=emb,
                threshold=threshold,
                margin=margin
            )

            enriched_results.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text,
                "speaker_id": decision["speaker_id"],
                "speaker_name": decision["display_name"],
                "role": decision["role"],
                "speaker_score": decision["score"],
                "status": decision["status"],
                "margin": decision["margin"],
                "provider": stt_provider or "sarvam"
            })

        return enriched_results

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
