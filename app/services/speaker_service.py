"""Speaker service — resolves speaker_id to participant identity.

This is the single source of truth for:
  speaker_id → display_name, role, user_id

Used by:
  - transcript_service (Phase 4) when assembling labelled transcript
  - transcript_analyzer (Phase 5) when building the Gemini prompt
  - Any router that needs to enrich a TranscriptSegment with names
"""

from app.database import get_db


def get_participant_by_speaker(meeting_id: str, speaker_id: str) -> dict | None:
    """
    Resolve a speaker_id to participant identity for a given meeting.

    Returns:
        {
            "speaker_id": "spk_001",
            "user_id": "manager_01",
            "display_name": "Amit",
            "role": "manager"
        }
        or None if not found.
    """
    with get_db() as conn:
        row = conn.execute(
            """SELECT speaker_id, user_id, display_name, role
               FROM meeting_participants
               WHERE meeting_id = ? AND speaker_id = ?""",
            (meeting_id, speaker_id),
        ).fetchone()

    if row is None:
        return None

    return {
        "speaker_id": row["speaker_id"],
        "user_id": row["user_id"],
        "display_name": row["display_name"],
        "role": row["role"],
    }


def get_all_participants(meeting_id: str) -> list[dict]:
    """
    Return all participants for a meeting, ordered by role (manager first).

    Returns list of dicts with keys: speaker_id, user_id, display_name, role
    """
    with get_db() as conn:
        rows = conn.execute(
            """SELECT speaker_id, user_id, display_name, role
               FROM meeting_participants
               WHERE meeting_id = ?
               ORDER BY CASE role WHEN 'manager' THEN 0 ELSE 1 END""",
            (meeting_id,),
        ).fetchall()

    return [
        {
            "speaker_id": row["speaker_id"],
            "user_id": row["user_id"],
            "display_name": row["display_name"],
            "role": row["role"],
        }
        for row in rows
    ]


def build_speaker_map(meeting_id: str) -> dict[str, dict]:
    """
    Build a lookup dict: speaker_id → participant info.

    Convenience wrapper for transcript_service use.

    Example return:
        {
            "spk_001": {"display_name": "Amit", "role": "manager", "user_id": "manager_01"},
            "spk_002": {"display_name": "Rahul", "role": "reportee", "user_id": "rep007"},
        }
    """
    participants = get_all_participants(meeting_id)
    return {
        p["speaker_id"]: {
            "display_name": p["display_name"],
            "role": p["role"],
            "user_id": p["user_id"],
        }
        for p in participants
    }


def register_participant(
    meeting_id: str,
    user_id: str,
    speaker_id: str,
    display_name: str,
    role: str,
) -> None:
    """
    Insert or update a participant row. Idempotent on (meeting_id, speaker_id).

    Args:
        meeting_id:   ID of the meeting
        user_id:      ID of the user (manager id or reportee id)
        speaker_id:   Speaker identifier assigned for this session, e.g. 'spk_001'
        display_name: Human-readable name for transcript display
        role:         'manager' or 'reportee'
    """
    with get_db() as conn:
        conn.execute(
            """INSERT INTO meeting_participants
               (meeting_id, user_id, speaker_id, display_name, role)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(meeting_id, speaker_id) DO UPDATE SET
                   user_id = excluded.user_id,
                   display_name = excluded.display_name,
                   role = excluded.role""",
            (meeting_id, user_id, speaker_id, display_name, role),
        )


def enrich_segments(
    segments: list,
    meeting_id: str,
) -> list:
    """
    Resolve speaker_id → display_name + role for every segment in-place.

    Fetches the speaker map once for the meeting, then stamps each segment
    that has a known speaker_id with speaker_name and role.

    Segments with no speaker_id or an unknown speaker_id are left unchanged
    (speaker_name and role remain None).

    Args:
        segments:   list[TranscriptSegment] from stt_service.transcribe_audio()
        meeting_id: ID of the meeting — used to look up meeting_participants

    Returns:
        The same list with speaker_name / role fields populated in-place.
        Returning the list makes call-site chaining convenient.
    """
    if not segments or not meeting_id:
        return segments

    speaker_map = build_speaker_map(meeting_id)
    if not speaker_map:
        return segments

    for seg in segments:
        if seg.speaker_id and seg.speaker_id in speaker_map:
            participant = speaker_map[seg.speaker_id]
            seg.speaker_name = participant["display_name"]
            seg.role = participant["role"]

    return segments
