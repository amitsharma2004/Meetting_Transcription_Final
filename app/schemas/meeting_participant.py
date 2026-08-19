"""Pydantic schemas for meeting participant endpoints."""

from typing import Literal
from pydantic import BaseModel, Field


class ParticipantCreate(BaseModel):
    """Request body for registering a participant in a meeting."""

    user_id: str = Field(..., description="ID of the user — manager id or reportee id")
    speaker_id: str = Field(..., description="WebRTC/STT speaker identifier, e.g. 'spk_001'")
    display_name: str = Field(..., description="Human-readable name shown in transcript UI")
    role: Literal["manager", "reportee"] = Field(
        ..., description="Role of this participant in the meeting"
    )


class ParticipantResponse(BaseModel):
    """Single participant entry returned by GET /meetings/{id}/participants."""

    speaker_id: str
    user_id: str
    display_name: str
    role: Literal["manager", "reportee"]


class ParticipantListResponse(BaseModel):
    """Response wrapper for participant list."""

    meeting_id: str
    participants: list[ParticipantResponse]
