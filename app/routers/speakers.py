"""Speaker Router — Endpoints for enrollment, profile inspection, participant management, and meeting RAM cache."""

import os
import shutil
import tempfile
import uuid
import re
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel

from app.services.speaker_profile_service import (
    enroll_speaker,
    enroll_meeting_participant,
    get_speaker_profile_from_db,
    load_meeting_cache,
    active_meetings
)
from app.services.speaker_service import get_all_participants
from app.schemas.meeting_participant import ParticipantListResponse, ParticipantResponse

router = APIRouter(prefix="", tags=["Speakers"])


class SpeakerProfileResponse(BaseModel):
    user_id: str
    speaker_id: str
    display_name: str
    role: str
    sample_count: int
    embedding_dimension: int
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@router.post("/speakers/enroll", status_code=status.HTTP_201_CREATED)
async def enroll_speaker_endpoint(
    user_id: str = Form(..., description="User ID (manager_01 or reportee ID)"),
    speaker_id: str = Form(..., description="Unique speaker ID (e.g. spk_001)"),
    display_name: str = Form(..., description="Speaker Name (e.g. Amit)"),
    role: str = Form("reportee", description="Role: 'manager' or 'reportee'"),
    audio_files: List[UploadFile] = File(..., description="2 to 5 clean audio sample WAV/WebM files")
):
    """
    Enrolls a speaker from 2-5 clean voice samples.
    Generates 192-dim normalized embeddings, master centroid, saves files to audio_samples/,
    and persists to SQLite.
    """
    if len(audio_files) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 audio samples required (3-5 recommended for robust centroid)."
        )

    # Save audio files to audio_samples/<speaker_id>/ for storage & reference
    samples_dir = Path(__file__).parent.parent.parent / "audio_samples" / speaker_id
    samples_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    try:
        for idx, file in enumerate(audio_files):
            file_ext = os.path.splitext(file.filename or "sample.wav")[1] or ".wav"
            file_path = str(samples_dir / f"sample_{idx+1}{file_ext}")
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            saved_paths.append(file_path)

        result = enroll_speaker(
            user_id=user_id,
            speaker_id=speaker_id,
            display_name=display_name,
            role=role,
            audio_sample_paths=saved_paths
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/speakers/{speaker_id}")
async def get_speaker_profile(speaker_id: str):
    """Retrieve enrolled profile from SQLite."""
    profile = get_speaker_profile_from_db(speaker_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Speaker profile '{speaker_id}' not found."
        )
    return profile


# ── Meeting Participant Endpoints ──────────────────────────────────────────

@router.get("/meetings/{meeting_id}/participants", response_model=ParticipantListResponse)
async def list_meeting_participants(meeting_id: str):
    """List all registered participants for a meeting session."""
    participants = get_all_participants(meeting_id)
    return ParticipantListResponse(
        meeting_id=meeting_id,
        participants=[
            ParticipantResponse(
                speaker_id=p["speaker_id"],
                user_id=p["user_id"],
                display_name=p["display_name"],
                role=p["role"],
            )
            for p in participants
        ],
    )


@router.post("/meetings/{meeting_id}/participants/enroll", status_code=status.HTTP_201_CREATED)
async def enroll_meeting_participant_endpoint(
    meeting_id: str,
    display_name: str = Form(..., description="Participant display name (e.g. Rahul)"),
    role: str = Form("reportee", description="Role: 'manager' or 'reportee'"),
    user_id: Optional[str] = Form(None, description="Optional user ID"),
    speaker_id: Optional[str] = Form(None, description="Optional custom speaker ID"),
    audio_files: List[UploadFile] = File(..., description="2 to 5 clean audio sample WAV/WebM files")
):
    """
    Combined Participant Voice Enrollment Endpoint:
    1. Saves uploaded samples to audio_samples/<speaker_id>/ directory.
    2. Extracts ECAPA-TDNN 192-dim embeddings on CPU.
    3. Computes Master Centroid vector.
    4. Saves to speaker_profiles and meeting_participants.
    5. Seamlessly updates in-memory RAM cache.
    """
    if not audio_files or len(audio_files) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 voice samples are required (3-5 recommended for robust centroid)."
        )

    role_clean = role.lower().strip()
    if role_clean not in {"manager", "reportee"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role '{role}'. Allowed roles are 'manager' or 'reportee'."
        )

    # Determine speaker_id
    if not speaker_id:
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '', display_name.lower().replace(' ', '_')) or "spk"
        speaker_id = f"spk_{clean_name}_{uuid.uuid4().hex[:4]}"

    # Save audio files into audio_samples/<speaker_id>/
    samples_dir = Path(__file__).parent.parent.parent / "audio_samples" / speaker_id
    samples_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    try:
        for idx, file in enumerate(audio_files):
            file_ext = os.path.splitext(file.filename or "sample.webm")[1] or ".webm"
            file_path = str(samples_dir / f"sample_{idx+1}{file_ext}")
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            saved_paths.append(file_path)

        result = enroll_meeting_participant(
            meeting_id=meeting_id,
            display_name=display_name.strip(),
            role=role_clean,
            audio_sample_paths=saved_paths,
            user_id=user_id.strip() if user_id else None,
            speaker_id=speaker_id,
        )
        return result

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Enrollment failed: {str(e)}")


# ── Meeting RAM Cache Endpoints ────────────────────────────────────────────

@router.post("/meetings/{meeting_id}/load-cache")
async def load_meeting_cache_endpoint(meeting_id: str):
    """Pre-loads meeting participants' centroids into in-memory RAM matrix."""
    try:
        cache_info = load_meeting_cache(meeting_id)
        return cache_info
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/meetings/{meeting_id}/clear-cache")
async def clear_meeting_cache_endpoint(meeting_id: str):
    """Clears meeting RAM cache on meeting end."""
    from app.services.speaker_profile_service import clear_meeting_cache
    return clear_meeting_cache(meeting_id)


@router.get("/meetings/{meeting_id}/cache-status")
async def get_meeting_cache_status(meeting_id: str):
    """Checks if a meeting is active in RAM cache."""
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
