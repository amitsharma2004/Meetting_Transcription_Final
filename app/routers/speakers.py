"""Speaker Router — Endpoints for enrollment, profile inspection, and meeting RAM cache."""

import os
import shutil
import tempfile
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel

from app.services.speaker_profile_service import (
    enroll_speaker,
    get_speaker_profile_from_db,
    load_meeting_cache,
    active_meetings
)

router = APIRouter(prefix="/speakers", tags=["Speakers"])


class SpeakerProfileResponse(BaseModel):
    user_id: str
    speaker_id: str
    display_name: str
    role: str
    sample_count: int
    embedding_dimension: int
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@router.post("/enroll", status_code=status.HTTP_201_CREATED)
async def enroll_speaker_endpoint(
    user_id: str = Form(..., description="User ID (manager_01 or reportee ID)"),
    speaker_id: str = Form(..., description="Unique speaker ID (e.g. spk_001)"),
    display_name: str = Form(..., description="Speaker Name (e.g. Amit)"),
    role: str = Form("reportee", description="Role: 'manager' or 'reportee'"),
    audio_files: List[UploadFile] = File(..., description="3 to 5 clean audio sample WAV files")
):
    """
    Step 2: Enrolls a speaker from 3-5 clean voice samples.
    Generates 192-dim normalized embeddings, master centroid, and persists to SQLite.
    """
    if len(audio_files) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 audio samples required (3-5 recommended for robust centroid)."
        )

    temp_dir = tempfile.mkdtemp(prefix="enroll_")
    temp_paths = []
    try:
        for idx, file in enumerate(audio_files):
            file_ext = os.path.splitext(file.filename or "sample.wav")[1] or ".wav"
            file_path = os.path.join(temp_dir, f"sample_{idx}{file_ext}")
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            temp_paths.append(file_path)

        result = enroll_speaker(
            user_id=user_id,
            speaker_id=speaker_id,
            display_name=display_name,
            role=role,
            audio_sample_paths=temp_paths
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("/{speaker_id}")
async def get_speaker_profile(speaker_id: str):
    """
    Step 3: Retrieve enrolled profile from SQLite (includes raw embeddings audit trail & centroid).
    """
    profile = get_speaker_profile_from_db(speaker_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Speaker profile '{speaker_id}' not found."
        )
    return profile


@router.post("/meetings/{meeting_id}/load-cache")
async def load_meeting_cache_endpoint(meeting_id: str):
    """
    Step 3: Pre-loads meeting participants' centroids from SQLite into in-memory RAM matrix.
    Enables sub-millisecond live matching without touching SQLite during live audio stream.
    """
    try:
        cache_info = load_meeting_cache(meeting_id)
        return cache_info
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/meetings/{meeting_id}/clear-cache")
async def clear_meeting_cache_endpoint(meeting_id: str):
    """
    Step 5: Clears meeting RAM cache on meeting end to free memory.
    """
    from app.services.speaker_profile_service import clear_meeting_cache
    return clear_meeting_cache(meeting_id)


@router.get("/meetings/{meeting_id}/cache-status")
async def get_meeting_cache_status(meeting_id: str):
    """Checks if a meeting is active in RAM cache and inspects cached participants."""
    if meeting_id not in active_meetings:
        return {"meeting_id": meeting_id, "cached": False}

    cache = active_meetings[meeting_id]
    return {
        "meeting_id": meeting_id,
        "cached": True,
        "speaker_ids": cache["speaker_ids"],
        "participant_count": len(cache["speaker_ids"]),
        "matrix_shape": list(cache["matrix"].shape)
    }
