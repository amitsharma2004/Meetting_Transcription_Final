"""STT Lab Router — Official SarvamAI SDK Batch (saaras:v4) and WebSocket (saaras:v3-realtime) Endpoints."""

import json
import asyncio
from fastapi import APIRouter, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from app.services.stt_lab_service import run_batch_stt, stream_pcm_to_sarvam

router = APIRouter(tags=["STT Lab"])


class BatchSTTResponse(BaseModel):
    mode: str = "batch"
    model: str = "saaras:v4"
    text: str
    job_id: Optional[str] = None
    diarized_entries: Optional[List[Dict[str, Any]]] = None
    with_diarization: Optional[bool] = False
    duration_seconds: Optional[float] = None
    filename: Optional[str] = None


@router.post("/stt-lab/batch", response_model=BatchSTTResponse, status_code=status.HTTP_200_OK)
async def batch_stt_endpoint(
    audio: UploadFile = File(..., description="Audio or Video file (.wav, .mp3, .mp4, .webm, .m4a)"),
    model: Optional[str] = Form("saaras:v4", description="Model: saaras:v4 or saaras:v3"),
    with_diarization: bool = Form(False, description="Enable Speaker Diarization (Multi-Speaker separation)")
):
    """
    Batch STT Endpoint using official SarvamAI SDK (saaras:v4 / saaras:v3):
    Receives an uploaded audio/video file, converts to 16kHz mono WAV,
    runs Sarvam Batch job (with optional speaker diarization), and returns full transcript.
    """
    if not audio:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio/Video file is required."
        )

    try:
        content = await audio.read()
        if len(content) < 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty or corrupted."
            )

        filename = audio.filename or "uploaded_media.wav"
        result = await run_batch_stt(
            input_bytes=content,
            filename=filename,
            model=model or "saaras:v4",
            with_diarization=with_diarization
        )
        return result

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Batch STT failed: {str(e)}")


# ── Realtime WebSocket Endpoint (saaras:v3-realtime) ──────────────────────────

@router.websocket("/ws/stt-lab")
async def websocket_stt_endpoint(websocket: WebSocket):
    """
    Realtime Speech-to-Text WebSocket endpoint using SarvamAI SDK (`saaras:v3-realtime`).
    Receives binary audio PCM/chunks from client browser and streams live transcript events.
    """
    await websocket.accept()
    audio_queue = asyncio.Queue()

    async def send_to_client(event: dict):
        try:
            await websocket.send_json(event)
        except Exception:
            pass

    # Start the Sarvam streaming worker
    sarvam_task = asyncio.create_task(
        stream_pcm_to_sarvam(audio_chunks_queue=audio_queue, event_callback=send_to_client)
    )

    try:
        while True:
            # Client sends binary PCM audio chunks or text control messages
            data = await websocket.receive()
            if "bytes" in data and data["bytes"]:
                await audio_queue.put(data["bytes"])
            elif "text" in data and data["text"]:
                msg = json.loads(data["text"])
                if msg.get("type") == "stop":
                    await audio_queue.put(None)  # End of stream
                    break

        await sarvam_task

    except WebSocketDisconnect:
        await audio_queue.put(None)
        if not sarvam_task.done():
            sarvam_task.cancel()
    except Exception as e:
        await audio_queue.put(None)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
