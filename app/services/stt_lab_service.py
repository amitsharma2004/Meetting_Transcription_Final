"""STT Lab Service — Official SarvamAI SDK Integration for Batch (saaras:v4) & Streaming (saaras:v3-realtime)."""

import os
import io
import json
import base64
import asyncio
import tempfile
import numpy as np
from typing import Tuple, Dict, Any, AsyncIterator
import av

from sarvamai import SarvamAI, AsyncSarvamAI, RealtimeAudioInput, RealtimeEnd
from app.config import settings


def decode_audio_to_pcm16k_wav(input_bytes: bytes, filename: str = "audio.wav") -> Tuple[bytes, float]:
    """
    Decodes arbitrary audio/video formats (mp4, webm, wav, mp3, m4a, ogg, etc.)
    and converts to standard 16kHz mono WAV bytes + duration in seconds.
    """
    input_buf = io.BytesIO(input_bytes)
    container = av.open(input_buf)

    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
    if not audio_stream:
        raise ValueError("No audio stream found in the uploaded file.")

    resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
    pcm_chunks = []

    for frame in container.decode(audio_stream):
        resampled_frames = resampler.resample(frame)
        for rf in resampled_frames:
            pcm_chunks.append(bytes(rf.planes[0]))

    raw_pcm = b"".join(pcm_chunks)
    container.close()

    if not raw_pcm:
        raise ValueError("Could not extract any audio data from file.")

    import wave
    out_buf = io.BytesIO()
    with wave.open(out_buf, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(raw_pcm)

    wav_bytes = out_buf.getvalue()
    duration_sec = len(raw_pcm) / (16000 * 2)
    return wav_bytes, float(duration_sec)


# ── 1. Batch STT Pipeline via SarvamAI SDK (saaras:v4) ──────────────────────────

async def run_batch_stt(
    input_bytes: bytes,
    filename: str = "audio.wav",
    model: str = "saaras:v4",
    with_diarization: bool = False
) -> Dict[str, Any]:
    """
    Runs asynchronous Batch STT using official SarvamAI SDK (saaras:v4 / saaras:v3)
    with optional speaker diarization and timestamps.
    """
    api_key = settings.SARVAM_API_KEY
    if not api_key:
        raise ValueError("SARVAM_API_KEY is not configured in settings or .env")

    # Normalize model name for Sarvam SDK
    target_model = "saaras:v4"
    if "v3" in model.lower():
        target_model = "saaras:v3"
    elif "v4" in model.lower():
        target_model = "saaras:v4"
    elif "saarika" in model.lower() or "v2.5" in model.lower():
        target_model = "saarika:v2.5"

    # 1. Convert to 16kHz mono WAV
    wav_bytes, duration_sec = decode_audio_to_pcm16k_wav(input_bytes, filename)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_wav.write(wav_bytes)
        tmp_path = tmp_wav.name

    out_dir = tempfile.mkdtemp(prefix="sarvam_batch_")

    try:
        def _execute_batch():
            client = SarvamAI(api_subscription_key=api_key)
            job_kwargs = {
                "model": target_model,
                "language_code": "en-IN",
                "with_timestamps": True,
            }
            if with_diarization:
                job_kwargs["with_diarization"] = True
            if target_model == "saaras:v3":
                job_kwargs["mode"] = "transcribe"

            job = client.speech_to_text_job.create_job(**job_kwargs)
            job.upload_files(file_paths=[tmp_path])
            job.start()
            status = job.wait_until_complete()
            job.download_outputs(output_dir=out_dir)
            return status, job.job_id

        loop = asyncio.get_event_loop()
        status, job_id = await loop.run_in_executor(None, _execute_batch)

        transcript_text = ""
        diarized_entries = []
        for fname in os.listdir(out_dir):
            if fname.endswith(".json"):
                with open(os.path.join(out_dir, fname), "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                    transcript_text = data.get("transcript", "").strip()
                    dt = data.get("diarized_transcript")
                    if dt and isinstance(dt, dict) and "entries" in dt:
                        diarized_entries = dt.get("entries", [])

        return {
            "mode": "batch_diarization" if with_diarization else "batch",
            "model": target_model,
            "job_id": job_id,
            "text": transcript_text,
            "diarized_entries": diarized_entries,
            "with_diarization": with_diarization,
            "duration_seconds": round(duration_sec, 2),
            "filename": filename
        }

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── 2. Realtime WebSocket Streaming Pipeline (saaras:v3-realtime) ─────────────

async def stream_pcm_to_sarvam(
    audio_chunks_queue: asyncio.Queue,
    event_callback
):
    """
    Connects to Sarvam realtime streaming WebSocket (`saaras:v3-realtime`)
    and pumps raw linear16 PCM audio chunks while broadcasting partial/final events.
    """
    api_key = settings.SARVAM_API_KEY
    if not api_key:
        raise ValueError("SARVAM_API_KEY is not configured in settings or .env")

    client = AsyncSarvamAI(api_subscription_key=api_key)

    async with client.speech_to_text_realtime_streaming.connect(
        language_code="en-IN",
        model="saaras:v3-realtime",
        mode="transcribe",
        stream_type="balanced",
    ) as ws:

        async def send_audio():
            while True:
                chunk = await audio_chunks_queue.get()
                if chunk is None:  # Sentinel value indicating end of stream
                    await ws.send_realtime_end(RealtimeEnd())
                    break
                
                b64_data = base64.b64encode(chunk).decode("utf-8")
                await ws.send_realtime_audio_input(RealtimeAudioInput(audio=b64_data))

        async def receive_events():
            seq = 0
            async for message in ws:
                seq += 1
                event_type = getattr(message, "event", "unknown")
                text = getattr(message, "text", "")

                if event_type == "transcript.partial":
                    await event_callback({
                        "type": "partial",
                        "event": event_type,
                        "seq": seq,
                        "text": text
                    })
                elif event_type == "transcript.final":
                    await event_callback({
                        "type": "final",
                        "event": event_type,
                        "seq": seq,
                        "text": text
                    })
                elif event_type in ("vad.speech_start", "vad.speech_end", "session.begin", "session.end"):
                    await event_callback({
                        "type": event_type.replace(".", "_"),
                        "event": event_type,
                        "seq": seq
                    })
                    if event_type == "session.end":
                        break
                elif event_type == "error":
                    await event_callback({
                        "type": "error",
                        "event": event_type,
                        "seq": seq,
                        "code": getattr(message, "code", "UNKNOWN"),
                        "message": getattr(message, "message", "Streaming error")
                    })
                    if getattr(message, "is_fatal", False):
                        break

        await asyncio.gather(send_audio(), receive_events())
