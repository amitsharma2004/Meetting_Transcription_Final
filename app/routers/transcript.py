"""Transcript router — POST /api/transcripts/analyze + /api/transcripts/transcribe."""

import json
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional

from app.database import get_db
from app.config import settings
from app.services.stt_service import transcribe_audio
from app.services.speaker_service import enrich_segments
from app.services.transcript_service import format_transcript, format_transcript_blocks
from app.services.live_recognition_service import transcribe_and_identify_speakers
from app.services.transcript_cleanup_service import cleanup_transcript, CleanTranscriptResult

router = APIRouter()


# ── STT endpoint ──────────────────────────────────────────────────────────────

@router.post("/transcripts/transcribe")
async def transcribe_audio_endpoint(
    audio: UploadFile = File(..., description="Audio file — wav, mp3, m4a, webm, ogg, flac"),
    meeting_id: Optional[str] = Form(None, description="Optional meeting ID for context"),
    speaker_id: Optional[str] = Form(None, description="Optional speaker ID to tag all segments"),
    analyze: bool = Form(False, description="If true, also run Gemini VED analysis on the labelled transcript"),
):
    """
    Transcribe an audio file using Whisper.

    Returns a list of timed transcript segments. Speaker name resolution
    happens in a later step (speaker_service, Phase 3).

    Supported formats: wav, mp3, m4a, webm, ogg, flac
    Max recommended size: 25 MB (larger files will be slow on CPU)
    """
    # Basic validation
    content_type = audio.content_type or ""
    allowed_types = {
        "audio/wav", "audio/wave", "audio/x-wav",
        "audio/mpeg", "audio/mp3",
        "audio/mp4", "audio/x-m4a",
        "audio/webm", "video/webm",
        "audio/ogg", "audio/flac",
        "application/octet-stream",   # browsers sometimes send this for blobs
    }
    if content_type and content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {content_type}. Send wav, mp3, m4a, webm, ogg, or flac.",
        )

    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    if len(audio_bytes) > 50 * 1024 * 1024:   # 50 MB hard limit
        raise HTTPException(status_code=413, detail="Audio file too large. Max 50 MB.")

    # Run Whisper
    segments = await transcribe_audio(audio_bytes, filename=audio.filename or "audio.wav")

    if not segments:
        raise HTTPException(
            status_code=422,
            detail="Whisper could not extract any speech from this audio file.",
        )

    # Attach speaker_id to every segment if provided
    if speaker_id:
        for seg in segments:
            seg.speaker_id = speaker_id

    # Resolve speaker_id → name + role via meeting_participants
    # Works only when meeting_id is given and participants are registered.
    # Segments with unresolvable speaker_id are returned as-is (graceful).
    if meeting_id:
        enrich_segments(segments, meeting_id)

    # Assemble labelled transcript string (ready for Gemini in Step 5)
    labelled = format_transcript(segments)

    # Optional: run Gemini VED analysis on the labelled transcript
    analysis = None
    if analyze:
        # Build participants dict from enriched segments for labelled mode
        participants = None
        if meeting_id:
            from app.services.speaker_service import build_speaker_map
            speaker_map = build_speaker_map(meeting_id)
            if speaker_map:
                manager_entry = next(
                    (v for v in speaker_map.values() if v["role"] == "manager"), None
                )
                reportee_entry = next(
                    (v for v in speaker_map.values() if v["role"] == "reportee"), None
                )
                if manager_entry or reportee_entry:
                    participants = {
                        "manager":  {"name": manager_entry["display_name"]} if manager_entry else {"name": "Unknown"},
                        "reportee": {"name": reportee_entry["display_name"]} if reportee_entry else {"name": "Unknown"},
                    }
        analysis = await analyze_transcript(labelled, participants=participants)

    response = {
        "meeting_id": meeting_id,
        "speaker_id": speaker_id,
        "segment_count": len(segments),
        "formatted_transcript": labelled,
        "segments": [seg.to_dict() for seg in segments],
    }
    if analysis is not None:
        response["analysis"] = analysis
    return response


# ── Formatted transcript endpoint ─────────────────────────────────────────────

@router.get("/meetings/{meeting_id}/transcript/formatted")
async def get_formatted_transcript(meeting_id: str):
    """
    Return the stored transcript for a meeting as a speaker-labelled string.

    This endpoint reconstructs a formatted transcript from:
      - meetings.transcript (raw text stored at meeting save time)
      - meeting_participants (speaker → name/role mapping)

    Since stored transcripts are plain text (not yet segmented with timestamps),
    this endpoint splits by newline and applies participant labels based on
    known speaker patterns in the text, or returns the raw transcript wrapped
    in a single speaker block if participants exist.

    Used by:
      - Frontend transcript viewer (human-readable view)
      - Step 5: Gemini analyzer input (labelled format)

    For meetings transcribed via /transcribe (with real segments + timestamps),
    the formatted_transcript field in that response is the preferred source.
    """
    with get_db() as conn:
        meeting = conn.execute(
            """SELECT m.transcript, r.first_name, r.last_name
               FROM meetings m
               JOIN reportees r ON m.reportee_id = r.id
               WHERE m.id = ?""",
            (meeting_id,),
        ).fetchone()

        if not meeting:
            raise HTTPException(status_code=404, detail="Meeting not found")

        raw_transcript = meeting["transcript"]
        if not raw_transcript or not raw_transcript.strip():
            raise HTTPException(
                status_code=404,
                detail="No transcript stored for this meeting.",
            )

        # Fetch participants for this meeting
        participants = conn.execute(
            """SELECT speaker_id, display_name, role
               FROM meeting_participants
               WHERE meeting_id = ?
               ORDER BY CASE role WHEN 'manager' THEN 0 ELSE 1 END""",
            (meeting_id,),
        ).fetchall()

    # Build participant info for context
    participant_info = [
        {"speaker_id": p["speaker_id"], "display_name": p["display_name"], "role": p["role"]}
        for p in participants
    ]

    return {
        "meeting_id": meeting_id,
        "participants": participant_info,
        "raw_transcript": raw_transcript,
        "formatted_transcript": raw_transcript,   # Step 5 will replace this with labelled version
        "note": "Stored transcripts are plain text. Use POST /transcripts/transcribe for real-time labelled output.",
    }


# Cache of recently processed chunk IDs: {(meeting_id, sequence)}
_processed_chunks_cache = set()

# ── Chunk ingestion endpoint ──────────────────────────────────────────────────

@router.post("/transcripts/chunk")
async def ingest_audio_chunk(
    audio: UploadFile = File(..., description="Audio chunk — wav, webm, ogg, mp4"),
    meeting_id: str = Form(..., description="Meeting ID — required for speaker resolution"),
    speaker_id: Optional[str] = Form(None, description="Speaker ID — e.g. spk_mgr_meet001 or 'auto' for dynamic ECAPA recognition"),
    sequence: int = Form(0, description="Chunk sequence number (for ordering)"),
    stt_provider: Optional[str] = Form("sarvam", description="STT Provider ('sarvam' | 'whisper' | 'web-speech')"),
):
    """
    Ingest a single audio chunk from MediaRecorder and return labelled segments.
    Includes sequence deduplication, duration calculation, STT provider routing, and dynamic ECAPA recognition.
    """
    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        return {"meeting_id": meeting_id, "speaker_id": speaker_id, "sequence": sequence, "segments": [], "formatted_transcript": ""}

    import hashlib
    chunk_hash = hashlib.md5(audio_bytes).hexdigest()
    if chunk_hash in _processed_chunks_cache:
        print(f"[CHUNK IGNORED - DUPLICATE] meeting={meeting_id} sequence={sequence} audio_hash={chunk_hash[:8]}")
        return {
            "meeting_id": meeting_id,
            "speaker_id": speaker_id or "auto",
            "sequence": sequence,
            "segment_count": 0,
            "formatted_transcript": "",
            "segments": [],
            "note": "duplicate chunk ignored"
        }
    _processed_chunks_cache.add(chunk_hash)
    if len(_processed_chunks_cache) > 5000:
        _processed_chunks_cache.clear()

    # 1. Automatic Dynamic Speaker Recognition Pipeline
    if not speaker_id or speaker_id.lower() in {"auto", "unknown", "none"}:
        try:
            results = await transcribe_and_identify_speakers(
                audio_bytes=audio_bytes,
                meeting_id=meeting_id,
                filename=audio.filename or "chunk.webm",
                threshold=getattr(settings, "SPEAKER_SIMILARITY_THRESHOLD", 0.40),
                margin=getattr(settings, "SPEAKER_SIMILARITY_MARGIN", 0.12),
                stt_provider=stt_provider
            )
            
            # Format text block
            lines = [f"[{r['start']}s - {r['end']}s] {r['speaker_name']} ({r['role']}): {r['text']}" for r in results]
            formatted_text = "\n".join(lines)
            
            return {
                "meeting_id": meeting_id,
                "speaker_id": "auto",
                "sequence": sequence,
                "segment_count": len(results),
                "formatted_transcript": formatted_text,
                "segments": results,
                "provider": stt_provider or "sarvam"
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Live recognition failed: {str(e)}")

    # 2. Pinned Speaker Pipeline
    try:
        segments = await transcribe_audio(
            audio_bytes,
            filename=audio.filename or "chunk.webm",
            provider=stt_provider
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"STT transcription failed: {str(e)}")

    if not segments:
        return {"meeting_id": meeting_id, "speaker_id": speaker_id, "sequence": sequence, "segments": [], "formatted_transcript": ""}

    for seg in segments:
        seg.speaker_id = speaker_id
    enrich_segments(segments, meeting_id)

    labelled = format_transcript(segments)

    return {
        "meeting_id": meeting_id,
        "speaker_id": speaker_id,
        "sequence": sequence,
        "segment_count": len(segments),
        "formatted_transcript": labelled,
        "segments": [seg.to_dict() for seg in segments],
    }


# ── Finalized Turn Cleanup Endpoint ──────────────────────────────────────────

class TranscriptCleanupRequest(BaseModel):
    raw_text: str
    speaker_name: Optional[str] = None
    speaker_id: Optional[str] = None
    language: Optional[str] = None


class TranscriptCleanupResponse(BaseModel):
    original_text: str
    cleaned_text: str
    changes_made: bool = False
    cleanup_status: str = "completed"


@router.post("/transcripts/cleanup", response_model=TranscriptCleanupResponse)
async def cleanup_transcript_endpoint(request: TranscriptCleanupRequest):
    """
    Cleans a finalized speaker turn: fixes transcription typos, stuttering, and punctuation
    WHILE strictly preserving language, code-mixing (Hinglish/Tamil), and technical terms.
    """
    if not request.raw_text or not request.raw_text.strip():
        return TranscriptCleanupResponse(
            original_text="",
            cleaned_text="",
            changes_made=False,
            cleanup_status="completed"
        )

    try:
        result = await cleanup_transcript(
            raw_text=request.raw_text,
            speaker_name=request.speaker_name,
            language=request.language
        )
        return TranscriptCleanupResponse(
            original_text=result.original_text,
            cleaned_text=result.cleaned_text,
            changes_made=result.changes_made,
            cleanup_status="completed"
        )
    except Exception as e:
        print(f"[transcript_cleanup_endpoint] Error during turn cleanup: {e}")
        # Graceful fallback: return raw text so transcript never breaks or disappears
        return TranscriptCleanupResponse(
            original_text=request.raw_text,
            cleaned_text=request.raw_text,
            changes_made=False,
            cleanup_status="failed"
        )
