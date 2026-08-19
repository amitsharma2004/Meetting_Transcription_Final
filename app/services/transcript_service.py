"""Transcript service — assembles enriched segments into a labelled transcript string.

Single responsibility: List[TranscriptSegment] → formatted string.

This service does NOT:
  - talk to the database
  - call Whisper
  - call Gemini

Those concerns belong to speaker_service, stt_service, and transcript_analyzer
respectively. This service is a pure formatting layer.

Typical pipeline position:
    stt_service.transcribe_audio()
        ↓
    speaker_service.enrich_segments()
        ↓
    transcript_service.format_transcript()   ← here
        ↓
    transcript_analyzer.analyze_transcript()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Any, Optional

if TYPE_CHECKING:
    from app.services.stt_service import TranscriptSegment


def format_transcript(segments: list) -> str:
    """
    Convert a list of enriched TranscriptSegments into a labelled transcript string.

    Each line format:
        [<start>] <Role> (<Name>): <text>

    If speaker_name is missing, falls back to speaker_id.
    If both are missing, uses "Unknown Speaker".
    If role is missing, omits the role prefix.

    Example output:
        [0.0] Manager (Arjun Mehta): Let us discuss the product roadmap.
        [2.5] Reportee (Priya Nair): I have completed the backend work.
        [5.0] Manager (Arjun Mehta): What is blocking the deployment?

    Args:
        segments: List of TranscriptSegment objects (from stt_service,
                  optionally enriched by speaker_service).

    Returns:
        Multi-line string suitable for feeding directly to Gemini
        transcript_analyzer. Empty string if segments list is empty.
    """
    if not segments:
        return ""

    lines: list[str] = []

    for seg in segments:
        timestamp = f"[{seg.start:.1f}]"

        # Build speaker label
        name = seg.speaker_name or seg.speaker_id or "Unknown Speaker"
        if seg.role:
            role_label = seg.role.capitalize()   # "manager" → "Manager"
            speaker_label = f"{role_label} ({name})"
        else:
            speaker_label = name

        lines.append(f"{timestamp} {speaker_label}: {seg.text}")

    return "\n".join(lines)


def format_transcript_blocks(segments: list) -> str:
    """
    Same as format_transcript() but groups consecutive segments from the same
    speaker into a single block — closer to how a human would read a transcript.

    Example output:
        [0.0] Manager (Arjun Mehta):
        Let us discuss the product roadmap. I think user research should come first.

        [5.0] Reportee (Priya Nair):
        I have completed the backend work. The database migration is blocking deployment.

        [12.0] Manager (Arjun Mehta):
        What is the biggest issue with the migration?

    Use this variant when the transcript is long and readability matters
    (e.g. for human review in the UI). Use format_transcript() for Gemini input.
    """
    if not segments:
        return ""

    blocks: list[str] = []
    current_speaker_id = None
    current_lines: list[str] = []
    current_label = ""
    current_start = 0.0

    for seg in segments:
        seg_speaker = seg.speaker_id or seg.speaker_name or "unknown"

        if seg_speaker != current_speaker_id:
            # Flush previous block
            if current_lines:
                block_text = " ".join(current_lines)
                blocks.append(f"[{current_start:.1f}] {current_label}:\n{block_text}")

            # Start new block
            current_speaker_id = seg_speaker
            current_start = seg.start
            name = seg.speaker_name or seg.speaker_id or "Unknown Speaker"
            if seg.role:
                current_label = f"{seg.role.capitalize()} ({name})"
            else:
                current_label = name
            current_lines = [seg.text]
        else:
            current_lines.append(seg.text)

    # Flush last block
    if current_lines:
        block_text = " ".join(current_lines)
        blocks.append(f"[{current_start:.1f}] {current_label}:\n{block_text}")

    return "\n\n".join(blocks)
