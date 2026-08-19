"""Transcript Cleanup Service.

Post-processes raw speech-to-text turns to fix punctuation, STT phonetic typos,
and stuttering artifacts WHILE strictly preserving language, code-mixing (Hinglish/Tamil),
informal tone, and technical terminology. Zero translation, zero summarization.
"""

import json
import logging
from typing import Optional, Dict, Any
import httpx
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)


class CleanTranscriptResult(BaseModel):
    """Structured response for transcript cleanup."""
    original_text: str = Field(..., description="Original raw STT utterance")
    cleaned_text: str = Field(..., description="Cleaned transcript utterance")
    changes_made: bool = Field(False, description="Whether any modifications were applied")


CLEANUP_SYSTEM_PROMPT = """You are a Specialized Real-Time Speech-to-Text Post-Processor and Transcript Cleaner for multilingual product and engineering meetings.

Your single task is to fix transcription artifacts and formatting errors in a single speech turn WITHOUT changing the meaning, language, or tone.

### STRICT RULES:

1. WHAT YOU MUST CORRECT:
   - Punctuation (commas, periods, question marks, exclamation marks).
   - Sentence-start and proper noun capitalization.
   - Obvious phonetic/acoustic STT transcription errors and typos (e.g., "this car sakte" -> "ye kar sakte", "cubernetes" -> "Kubernetes", "rediss" -> "Redis").
   - Accidental repeated words caused by speech stuttering or STT loops (e.g., "we we deployed" -> "we deployed", "hum hum" -> "hum").
   - Obvious grammatical artifacts caused by bad voice transcription.

2. NEVER TRANSLATE OR CHANGE LANGUAGE (CRITICAL):
   - DO NOT translate Hindi to English.
   - DO NOT translate Tamil to English.
   - DO NOT translate English to Hindi.
   - DO NOT translate Romanized Hinglish (e.g., "haan basically hum ye kar sakte hain" MUST REMAIN "Haan, basically hum ye kar sakte hain").
   - Preserve Roman/Latin script if input is in Roman script.
   - Preserve regional vocabulary, slang, code-switching, and informal speech.

3. PRESERVE TECHNICAL TERMS & NAMES:
   - Preserve software names, technical terms, frameworks, and tools (e.g. Kubernetes, Redis, Docker, FastAPI, Python, AWS, ECAPA, PostgreSQL, React, GraphQL, Kafka, etc.).
   - Preserve person names and project names.

4. NEVER PARAPHRASE OR SUMMARIZE:
   - Do NOT rewrite or condense sentences.
   - Do NOT formalize casual speech.
   - Do NOT add new information or remove meaningful context.

5. OUTPUT FORMAT:
   - Output MUST be a valid JSON object matching this exact schema:
     {
       "original_text": "<exact input text>",
       "cleaned_text": "<cleaned text>",
       "changes_made": true/false
     }
"""


async def cleanup_transcript(
    raw_text: str,
    speaker_name: Optional[str] = None,
    language: Optional[str] = None,
) -> CleanTranscriptResult:
    """
    Cleans raw STT text turn using fast LLM post-processing with strict structured validation.
    
    Args:
        raw_text: The raw transcribed speech string.
        speaker_name: Optional name of the speaker for context.
        language: Optional language hint (e.g., 'hi-IN', 'en-IN', 'ta-IN', 'unknown').
        
    Returns:
        CleanTranscriptResult with original_text, cleaned_text, and changes_made.
    """
    clean_input = (raw_text or "").strip()
    if not clean_input:
        return CleanTranscriptResult(original_text="", cleaned_text="", changes_made=False)

    # Fast heuristic skip for single trivial words
    if len(clean_input.split()) == 1 and clean_input.lower() in {"yes", "no", "okay", "haan", "theek", "sure"}:
        return CleanTranscriptResult(
            original_text=clean_input,
            cleaned_text=clean_input.capitalize() + ".",
            changes_made=clean_input != clean_input.capitalize() + "."
        )

    # Build prompt context
    user_payload: Dict[str, Any] = {"raw_text": clean_input}
    if speaker_name:
        user_payload["speaker_name"] = speaker_name
    if language and language != "unknown":
        user_payload["language_hint"] = language

    user_message = f"Clean the following raw transcript turn:\n```json\n{json.dumps(user_payload, ensure_ascii=False)}\n```"

    # Attempt LLM Cleanup via Groq (fastest) or Gemini fallback
    groq_api_key = settings.GROQ_API_KEY
    model_name = settings.GROQ_MODEL or "openai/gpt-oss-120b"

    if groq_api_key:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {groq_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": CLEANUP_SYSTEM_PROMPT},
                            {"role": "user", "content": user_message},
                        ],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.1,
                    },
                )

                if res.status_code == 200:
                    data = res.json()
                    content = data["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    cleaned_str = parsed.get("cleaned_text", clean_input).strip()
                    
                    return CleanTranscriptResult(
                        original_text=clean_input,
                        cleaned_text=cleaned_str,
                        changes_made=bool(parsed.get("changes_made", cleaned_str != clean_input))
                    )
                else:
                    logger.warning(f"[cleanup_transcript] Groq API returned status {res.status_code}: {res.text}")
        except Exception as e:
            logger.error(f"[cleanup_transcript] Groq error: {e}", exc_info=True)

    # Fallback to direct heuristic / original text if LLM unavailable
    return CleanTranscriptResult(
        original_text=clean_input,
        cleaned_text=clean_input,
        changes_made=False
    )
