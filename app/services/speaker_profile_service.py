"""Speaker Profile Service — Enrollment, Master Centroid, Persistence, and RAM Cache.

Implements:
  - Step 2: Audio samples -> Individual 192-dim Embeddings -> Master Normalized Centroid
  - Step 3: SQLite storage (Audit raw embeddings + Centroid vector)
  - Step 3: In-Memory Meeting RAM Cache for sub-millisecond live matching without DB queries
"""

import json
import os
import re
import uuid
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Any

from app.database import get_db

# ---------------------------------------------------------------------------
# Global Model Singleton & In-Memory RAM Cache
# ---------------------------------------------------------------------------
_MODEL_INSTANCE = None

# Runtime Cache Structure:
# active_meetings = {
#     "meet_001": {
#         "speaker_ids": ["spk_001", "spk_002"],
#         "metadata": {
#             "spk_001": {"name": "Amit", "role": "manager", "user_id": "mgr_01"},
#             "spk_002": {"name": "Rahul", "role": "reportee", "user_id": "rep_01"}
#         },
#         "matrix": np.ndarray (K, 192)  # Pre-stacked for instant dot-product
#     }
# }
active_meetings: Dict[str, Dict[str, Any]] = {}


def get_ecapa_model():
    """Singleton loader for ECAPA-TDNN model on CPU."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier

        from pathlib import Path
        model_dir = str(Path(__file__).parent.parent.parent / "pretrained_models" / "spkrec-ecapa-voxceleb")

        _MODEL_INSTANCE = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=model_dir,
            run_opts={"device": "cpu"}
        )
    return _MODEL_INSTANCE


# ---------------------------------------------------------------------------
# STEP 2: Enrollment Pipeline (Audio -> Embeddings -> Centroid)
# ---------------------------------------------------------------------------
def extract_normalized_embedding(audio_path_or_tensor) -> np.ndarray:
    """Extracts 192-dim L2-normalized embedding on CPU."""
    model = get_ecapa_model()
    
    if isinstance(audio_path_or_tensor, str):
        signal = model.load_audio(audio_path_or_tensor)
    elif isinstance(audio_path_or_tensor, bytes):
        import av
        import io
        container = av.open(io.BytesIO(audio_path_or_tensor))
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=16000)
        frames = []
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                frames.append(resampled.to_ndarray())
        container.close()
        pcm = np.concatenate([f.squeeze() for f in frames]).astype(np.float32) if frames else np.zeros((1600,), dtype=np.float32)
        signal = torch.as_tensor(pcm, dtype=torch.float32).unsqueeze(0)
    elif isinstance(audio_path_or_tensor, (np.ndarray, torch.Tensor)):
        signal = torch.as_tensor(audio_path_or_tensor, dtype=torch.float32)
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported audio input type: {type(audio_path_or_tensor)}")

    with torch.no_grad():
        emb = model.encode_batch(signal).squeeze()
        emb = F.normalize(emb, p=2, dim=0)
    return emb.cpu().numpy()


def compute_master_centroid(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Computes L2-normalized centroid from multiple raw voice sample embeddings.
    centroid = normalize(mean(e1, e2, ..., eN))
    """
    if not embeddings:
        raise ValueError("Cannot compute centroid from empty embeddings list")

    # 1. Arithmetic mean across all sample vectors
    mean_vec = np.mean(embeddings, axis=0)

    # 2. L2 normalize master centroid
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        centroid = mean_vec / norm
    else:
        centroid = mean_vec

    return centroid.astype(np.float32)


def enroll_speaker(
    user_id: str,
    speaker_id: str,
    display_name: str,
    role: str,
    audio_sample_paths: List[str]
) -> dict:
    """
    Step 2 Core: Enrolls speaker from 3-5 audio clips, calculates raw embeddings + centroid,
    and persists into SQLite.
    """
    if len(audio_sample_paths) < 2:
        raise ValueError("Enrollment requires at least 2 clean audio samples (3-5 recommended).")

    raw_embeddings = []
    for path in audio_sample_paths:
        emb = extract_normalized_embedding(path)
        raw_embeddings.append(emb)

    # Compute master centroid
    centroid = compute_master_centroid(raw_embeddings)

    # Step 3: Save to SQLite (both individual embeddings for audit & centroid)
    save_speaker_profile_to_db(
        user_id=user_id,
        speaker_id=speaker_id,
        display_name=display_name,
        role=role,
        raw_embeddings=raw_embeddings,
        centroid=centroid
    )

    return {
        "user_id": user_id,
        "speaker_id": speaker_id,
        "display_name": display_name,
        "role": role,
        "sample_count": len(raw_embeddings),
        "embedding_dimension": 192,
        "status": "ENROLLED"
    }


def enroll_meeting_participant(
    meeting_id: str,
    display_name: str,
    role: str = "reportee",
    audio_sample_paths: List[str] = None,
    user_id: Optional[str] = None,
    speaker_id: Optional[str] = None,
) -> dict:
    """
    Combined Enrollment Flow:
    1. Validates inputs & role.
    2. Auto-generates user_id / speaker_id if not supplied.
    3. Extracts ECAPA embeddings and computes Master Centroid.
    4. Saves to speaker_profiles (audit raw embeddings + centroid).
    5. Saves/links to meeting_participants for meeting_id.
    6. Returns structured response with status 'VOICE_READY'.
    """
    if not audio_sample_paths or len(audio_sample_paths) < 2:
        raise ValueError("At least 2 audio samples required (3-5 recommended for robust centroid).")

    role_clean = role.lower().strip()
    if role_clean not in {"manager", "reportee"}:
        raise ValueError(f"Invalid role '{role}'. Must be 'manager' or 'reportee'.")

    # Generate user_id if omitted
    if not user_id:
        user_id = f"user_{uuid.uuid4().hex[:6]}"

    # Generate speaker_id if omitted
    if not speaker_id:
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '', display_name.lower().replace(' ', '_')) or "spk"
        speaker_id = f"spk_{clean_name}_{uuid.uuid4().hex[:4]}"

    # 1. Extract embeddings
    raw_embeddings = []
    for sample in audio_sample_paths:
        emb = extract_normalized_embedding(sample)
        raw_embeddings.append(emb)

    # 2. Compute master centroid
    centroid = compute_master_centroid(raw_embeddings)

    # 3. Save to speaker_profiles (permanent voice store)
    save_speaker_profile_to_db(
        user_id=user_id,
        speaker_id=speaker_id,
        display_name=display_name,
        role=role_clean,
        raw_embeddings=raw_embeddings,
        centroid=centroid
    )

    # 4. Save to meeting_participants (session mapping)
    with get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO meetings (id, meeting_date) VALUES (?, CURRENT_DATE)",
            (meeting_id,)
        )

        conn.execute(
            """INSERT INTO meeting_participants
               (meeting_id, user_id, speaker_id, display_name, role)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(meeting_id, speaker_id) DO UPDATE SET
                   user_id = excluded.user_id,
                   display_name = excluded.display_name,
                   role = excluded.role""",
            (meeting_id, user_id, speaker_id, display_name, role_clean),
        )

    # 5. If meeting is currently cached in RAM, update RAM cache seamlessly
    if meeting_id in active_meetings:
        try:
            load_meeting_cache(meeting_id)
        except Exception:
            pass

    return {
        "meeting_id": meeting_id,
        "speaker_id": speaker_id,
        "user_id": user_id,
        "display_name": display_name,
        "role": role_clean,
        "sample_count": len(raw_embeddings),
        "embedding_dimension": 192,
        "profile_created": True,
        "participant_added": True,
        "status": "VOICE_READY"
    }


# ---------------------------------------------------------------------------
# STEP 3: SQLite Persistence & Retrieval
# ---------------------------------------------------------------------------
def save_speaker_profile_to_db(
    user_id: str,
    speaker_id: str,
    display_name: str,
    role: str,
    raw_embeddings: List[np.ndarray],
    centroid: np.ndarray
) -> None:
    """Saves raw embeddings list and centroid as clean JSON in SQLite."""
    embeddings_list = [emb.tolist() for emb in raw_embeddings]
    centroid_list = centroid.tolist()

    with get_db() as conn:
        conn.execute(
            """INSERT INTO speaker_profiles 
               (user_id, speaker_id, display_name, role, embedding_dimension, 
                sample_count, embeddings_json, centroid_json, updated_at)
               VALUES (?, ?, ?, ?, 192, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(speaker_id) DO UPDATE SET
                   user_id = excluded.user_id,
                   display_name = excluded.display_name,
                   role = excluded.role,
                   sample_count = excluded.sample_count,
                   embeddings_json = excluded.embeddings_json,
                   centroid_json = excluded.centroid_json,
                   updated_at = CURRENT_TIMESTAMP""",
            (
                user_id,
                speaker_id,
                display_name,
                role,
                len(raw_embeddings),
                json.dumps(embeddings_list),
                json.dumps(centroid_list),
            )
        )


def get_speaker_profile_from_db(speaker_id: str) -> Optional[dict]:
    """Retrieves full profile including raw embeddings for audit/debugging."""
    with get_db() as conn:
        row = conn.execute(
            """SELECT user_id, speaker_id, display_name, role, sample_count, 
                      embeddings_json, centroid_json, created_at, updated_at
               FROM speaker_profiles WHERE speaker_id = ?""",
            (speaker_id,)
        ).fetchone()

    if not row:
        return None

    return {
        "user_id": row["user_id"],
        "speaker_id": row["speaker_id"],
        "display_name": row["display_name"],
        "role": row["role"],
        "sample_count": row["sample_count"],
        "embeddings": json.loads(row["embeddings_json"]),
        "centroid": json.loads(row["centroid_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"]
    }


# ---------------------------------------------------------------------------
# STEP 3: In-Memory Meeting RAM Cache
# ---------------------------------------------------------------------------
def load_meeting_cache(meeting_id: str) -> dict:
    """
    Loads participants' Centroids AND all individual sample embeddings into RAM.
    Zero-DB calls during live streaming!
    """
    with get_db() as conn:
        # Join meeting_participants with speaker_profiles
        rows = conn.execute(
            """SELECT mp.speaker_id, mp.user_id, mp.display_name, mp.role, sp.centroid_json, sp.embeddings_json
               FROM meeting_participants mp
               JOIN speaker_profiles sp ON mp.speaker_id = sp.speaker_id
               WHERE mp.meeting_id = ?""",
            (meeting_id,)
        ).fetchall()

    if not rows:
        raise ValueError(f"No enrolled participants found for meeting '{meeting_id}'")

    speaker_ids = []
    metadata = {}
    centroids = []
    speaker_samples = {}

    for r in rows:
        spk_id = r["speaker_id"]
        speaker_ids.append(spk_id)
        metadata[spk_id] = {
            "user_id": r["user_id"],
            "name": r["display_name"],
            "role": r["role"]
        }
        cent_arr = json.loads(r["centroid_json"])
        centroids.append(cent_arr)

        raw_list = json.loads(r["embeddings_json"]) if r["embeddings_json"] else []
        if raw_list:
            speaker_samples[spk_id] = np.array(raw_list, dtype=np.float32)
        else:
            speaker_samples[spk_id] = np.array([cent_arr], dtype=np.float32)

    # Stack centroids into a single (K, 192) matrix for instant vector dot product
    matrix = np.array(centroids, dtype=np.float32)

    active_meetings[meeting_id] = {
        "speaker_ids": speaker_ids,
        "metadata": metadata,
        "matrix": matrix,
        "speaker_samples": speaker_samples,
        "temporal_state": {
            "last_speaker_id": None,
            "last_speaker_score": 0.0,
            "streak": 0
        }
    }

    return {
        "meeting_id": meeting_id,
        "participant_count": len(speaker_ids),
        "speakers": [metadata[sid]["name"] for sid in speaker_ids]
    }


def clear_meeting_cache(meeting_id: str) -> dict:
    """
    Clears meeting participants' matrix and sample vectors from RAM when meeting ends.
    """
    removed = active_meetings.pop(meeting_id, None)
    return {
        "meeting_id": meeting_id,
        "cleared": removed is not None,
        "remaining_active_meetings": len(active_meetings)
    }


def match_live_embedding_in_cache(
    meeting_id: str,
    live_embedding: np.ndarray,
    threshold: float = 0.40,
    margin: float = 0.12,
    enable_temporal_smoothing: bool = True
) -> dict:
    """
    Hybrid Dual-Scoring Matcher:
    1. Computes Master Centroid score for each enrolled speaker.
    2. Computes Multi-Sample scores (Top-1 and Top-2 average) across all enrolled voiceprints.
    3. Fused candidate score S_i = max(S_centroid, S_top2_sample)
    4. Evaluates runner-up margin: Δ = S_1 - S_2 ≥ margin
    5. Applies Temporal Continuity Smoothing for continuous speech turns to eliminate single-chunk Unknown dropouts.
    """
    if meeting_id not in active_meetings:
        raise KeyError(f"Meeting '{meeting_id}' is not loaded in RAM cache. Call load_meeting_cache first.")

    cache = active_meetings[meeting_id]
    matrix = cache["matrix"]           # Shape (K, 192)
    speaker_ids = cache["speaker_ids"] # Length K
    metadata = cache["metadata"]
    speaker_samples = cache.get("speaker_samples", {})
    temporal_state = cache.setdefault("temporal_state", {
        "last_speaker_id": None,
        "last_speaker_score": 0.0,
        "streak": 0
    })

    # Normalize live embedding to unit hypersphere
    norm = np.linalg.norm(live_embedding)
    if norm > 0:
        live_embedding = live_embedding / norm

    # 1. Instant Centroid Matrix Dot-Product
    centroid_scores = np.dot(matrix, live_embedding)

    # 2. Compute Hybrid Fusion Score for each enrolled candidate
    candidate_scores = []
    for idx, spk_id in enumerate(speaker_ids):
        cent_score = float(centroid_scores[idx])

        # Multi-sample comparison
        samples = speaker_samples.get(spk_id)
        if samples is not None and len(samples) > 0:
            sample_sims = np.dot(samples, live_embedding)
            top1_sample = float(np.max(sample_sims))
            if len(sample_sims) >= 2:
                top2_sample = float(np.mean(np.partition(sample_sims, -2)[-2:]))
            else:
                top2_sample = top1_sample

            # Fusion: take the best representation of this speaker (Centroid vs Top-2 Average)
            fused_score = max(cent_score, top2_sample)
        else:
            top1_sample = cent_score
            top2_sample = cent_score
            fused_score = cent_score

        candidate_scores.append({
            "speaker_id": spk_id,
            "centroid_score": cent_score,
            "top1_sample_score": top1_sample,
            "top2_sample_score": top2_sample,
            "fused_score": fused_score,
            "metadata": metadata[spk_id]
        })

    # 3. Sort candidates by fused_score descending
    candidate_scores.sort(key=lambda c: c["fused_score"], reverse=True)
    best_candidate = candidate_scores[0]
    best_score = best_candidate["fused_score"]
    best_speaker_id = best_candidate["speaker_id"]
    best_meta = best_candidate["metadata"]

    # Runner-up score and margin
    if len(candidate_scores) > 1:
        second_candidate = candidate_scores[1]
        second_score = second_candidate["fused_score"]
    else:
        second_score = -1.0

    score_diff = best_score - second_score

    # 4. Temporal Smoothing Prior:
    # If the candidate was already actively speaking in the immediate previous chunk (last_speaker_id == best_speaker_id)
    # and the current chunk has score near threshold [threshold - 0.05, threshold) with clear margin over runner-up:
    # Apply soft continuity prior to prevent 1-chunk Unknown flicker during low-energy speech/pauses.
    is_temporally_boosted = False
    if enable_temporal_smoothing and temporal_state.get("last_speaker_id") == best_speaker_id:
        if (threshold - 0.05) <= best_score < threshold and score_diff >= margin:
            best_score = threshold + 0.02
            is_temporally_boosted = True

    # 5. Unknown Speaker Gate (Below Threshold)
    if best_score < threshold:
        temporal_state["last_speaker_id"] = None
        temporal_state["streak"] = 0
        return {
            "status": "UNKNOWN",
            "speaker_id": None,
            "display_name": "Unknown Speaker",
            "role": "guest",
            "score": round(best_candidate["fused_score"], 4),
            "centroid_score": round(best_candidate["centroid_score"], 4),
            "sample_score": round(best_candidate["top2_sample_score"], 4),
            "second_score": round(second_score, 4),
            "margin": round(score_diff, 4),
            "raw_best_match": best_meta["name"]
        }

    # 6. Ambiguity / Overlap Protection (Too close to second speaker)
    if len(candidate_scores) > 1 and score_diff < margin:
        return {
            "status": "UNCERTAIN",
            "speaker_id": best_speaker_id,
            "display_name": f"{best_meta['name']} (Uncertain)",
            "role": best_meta["role"],
            "score": round(best_score, 4),
            "centroid_score": round(best_candidate["centroid_score"], 4),
            "sample_score": round(best_candidate["top2_sample_score"], 4),
            "second_score": round(second_score, 4),
            "margin": round(score_diff, 4)
        }

    # 7. Clean Matched Speaker
    temporal_state["last_speaker_id"] = best_speaker_id
    temporal_state["last_speaker_score"] = best_score
    temporal_state["streak"] = temporal_state.get("streak", 0) + 1

    return {
        "status": "MATCHED",
        "speaker_id": best_speaker_id,
        "display_name": best_meta["name"],
        "role": best_meta["role"],
        "user_id": best_meta.get("user_id"),
        "score": round(best_score, 4),
        "centroid_score": round(best_candidate["centroid_score"], 4),
        "sample_score": round(best_candidate["top2_sample_score"], 4),
        "second_score": round(second_score, 4),
        "margin": round(score_diff, 4),
        "streak": temporal_state["streak"],
        "temporally_boosted": is_temporally_boosted
    }


def recognize_speaker_from_audio(
    meeting_id: str,
    audio_path_or_array,
    threshold: float = 0.50,
    margin: float = 0.15
) -> dict:
    """
    End-to-End recognition: Audio -> ECAPA embedding -> RAM matrix match.
    """
    emb = extract_normalized_embedding(audio_path_or_array)
    return match_live_embedding_in_cache(
        meeting_id=meeting_id,
        live_embedding=emb,
        threshold=threshold,
        margin=margin
    )

