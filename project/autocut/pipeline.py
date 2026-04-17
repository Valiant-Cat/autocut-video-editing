from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from config import Settings, load_settings
from llm_clients import build_clients, extract_json_block
from media import (
    build_candidate_segments,
    cap_candidates,
    detect_scene_changes,
    extract_frame_samples,
    extract_summary_video_chunk,
    ffprobe_duration,
    ffprobe_video_fps,
    render_timeline,
    transcode_audio_for_analysis,
    write_json,
)


WHOLE_VIDEO_SUMMARY_PROMPT_TEMPLATE = """
You are analyzing the full source video before automatic editing.

Video duration: {video_duration:.2f} seconds.

Return strict JSON:
{{
 "summary": "what this video is broadly about",
 "story_overview": "short story or content overview",
 "main_characters": ["..."],
 "key_themes": ["..."],
 "tone_arc": "how the emotional tone evolves",
 "emotional_peaks": [
  {{
   "description": "peak description",
   "emotion": "emotion label",
   "importance": "low|medium|high"
  }}
 ],
 "mixing_guidance": "when original dialogue or native sound should be emphasized"
}}

Rules:
- Focus on narrative, topic flow, emotional changes, and moments that would benefit from keeping source audio.
- Output JSON only.
""".strip()


WHOLE_VIDEO_CHUNK_PROMPT_TEMPLATE = """
You are analyzing one chunk from a longer source video before automatic editing.

Chunk range: {start_sec:.2f}s - {end_sec:.2f}s.
Chunk duration: {duration:.2f} seconds.

Return strict JSON:
{{
 "chunk_summary": "one concise summary",
 "story_progress": "what happens in this chunk",
 "characters": ["..."],
 "key_events": ["..."],
 "dominant_emotion": "short phrase",
 "source_audio_moments": ["moments where original dialogue/reaction/ambience should be heard"],
 "highlight_level": "low|medium|high"
}}

Rules:
- Focus on plot movement, emotional change, and moments that may need source audio emphasis.
- Keep items concise.
- Output JSON only.
""".strip()


WHOLE_VIDEO_AGGREGATE_PROMPT_TEMPLATE = """
You are merging chunk-level analyses into one final understanding of the entire source video.

Video duration: {video_duration:.2f} seconds.
Chunk analyses:
{chunk_summaries}

Return strict JSON:
{{
 "summary": "what this video is broadly about",
 "story_overview": "short story or content overview",
 "main_characters": ["..."],
 "key_themes": ["..."],
 "tone_arc": "how the emotional tone evolves",
 "emotional_peaks": [
  {{
   "description": "peak description with approximate time range if possible",
   "emotion": "emotion label",
   "importance": "low|medium|high"
  }}
 ],
 "mixing_guidance": "when original dialogue or native sound should be emphasized"
}}

Rules:
- Base the answer only on the chunk analyses provided.
- Merge duplicate characters, events, and themes.
- Prefer concise but informative output.
- Output JSON only.
""".strip()


VIDEO_COLLECTION_SUMMARY_PROMPT_TEMPLATE = """
You are merging multiple source-video summaries into one combined understanding for remix editing.

Total source duration: {video_duration:.2f} seconds.
Source summaries:
{source_summaries}

Return strict JSON:
{{
 "summary": "what the full material pool is broadly about",
 "story_overview": "how the source videos together can be understood or remixed",
 "main_characters": ["..."],
 "key_themes": ["..."],
 "tone_arc": "overall emotional progression across the material pool",
 "emotional_peaks": [
  {{
   "description": "peak description with source reference if useful",
   "emotion": "emotion label",
   "importance": "low|medium|high"
  }}
 ],
 "mixing_guidance": "when original dialogue or native sound should be emphasized"
}}

Rules:
- Base the answer only on the source summaries provided.
- Treat the materials as a combined remix pool.
- Output JSON only.
""".strip()


SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


VIDEO_ANALYSIS_PROMPT_TEMPLATE = """
You are analyzing a candidate video clip for automatic editing.

Whole video context:
{whole_video_context}

Return strict JSON:
{{
 "summary": "one concise sentence",
 "subjects": ["..."],
 "actions": ["..."],
 "mood": "short phrase",
 "visual_tags": ["..."],
 "edit_use": "why this clip is useful",
 "suitability_score": 1,
 "emotional_intensity": 1,
 "source_audio_priority": false,
 "narrative_role": "what this clip contributes"
}}

Rules:
- Base your answer only on the provided frames, but use the whole-video context to better understand narrative meaning.
- emotional_intensity must be an integer from 1 to 5.
- source_audio_priority should be true when this clip is likely better if original dialogue, shout, reaction, or native ambience is heard clearly.
- Keep tags short.
- Output JSON only.
""".strip()


AUDIO_ANALYSIS_PROMPT_TEMPLATE = """
You are analyzing soundtrack structure for automatic video editing.

Audio duration: {audio_duration:.2f} seconds.

Return strict JSON:
{{
 "summary": "overall music summary",
 "sections": [
  {{
   "start_sec": 0.0,
   "end_sec": 0.0,
   "mood": "short phrase",
   "energy": "low|medium|high",
   "edit_purpose": "how visuals should feel here"
  }}
 ]
}}

Rules:
- Sections must cover the whole audio timeline in order.
- Use 3 to 8 sections.
- Keep timestamps numeric in seconds.
- Output JSON only.
""".strip()


PLANNER_SYSTEM_PROMPT = """
You are a senior video editor.
Your job is to choose source clips for each audio slot so the final edit matches the user prompt and the music arc.
Return JSON only.
""".strip()


def _log(message: str) -> None:
    print(message, flush=True)


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", value).strip().replace(" ", "_")
    cleaned = cleaned[:48] or "job"
    suffix = hashlib.md5(value.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned}_{suffix}"


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _resolve_parallelism(limit: int, task_count: int) -> tuple[int, int]:
    configured_limit = max(1, int(limit))
    actual_workers = max(1, min(configured_limit, max(1, int(task_count))))
    return configured_limit, actual_workers


def _resolve_source_videos(video_path: str | Path) -> list[Path]:
    resolved = Path(video_path).expanduser().resolve()
    if resolved.is_file():
        return [resolved]
    if not resolved.is_dir():
        raise FileNotFoundError(f"Video input not found: {resolved}")
    videos = sorted(
        [path for path in resolved.rglob('*') if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES]
    )
    if not videos:
        raise RuntimeError(f"No supported video files found in directory: {resolved}")
    return videos


def _to_relative_path(path: str | Path, base_dir: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return os.path.relpath(str(resolved), str(base_dir))
    except ValueError:
        return str(resolved)


def _with_relative_source_paths(items: list[dict[str, Any]], base_dir: Path) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in items:
        updated = dict(item)
        if "source_video_path" in updated:
            updated["source_video_path"] = _to_relative_path(str(updated["source_video_path"]), base_dir)
        normalized.append(updated)
    return normalized


def _summarize_video_collection(
    source_videos: list[dict[str, Any]],
    video_client,
    settings: Settings,
    work_dir: Path,
) -> dict[str, Any]:
    if len(source_videos) == 1:
        source = source_videos[0]
        return _summarize_whole_video(str(source["path"]), float(source["duration"]), video_client, settings, work_dir / source["source_id"])

    _log(f"Starting full source understanding for {len(source_videos)} videos")
    summaries: list[dict[str, Any]] = []
    for source in source_videos:
        _log(f"  Understanding source {source['source_id']}: {source['path'].name}")
        summary = _summarize_whole_video(
            str(source["path"]),
            float(source["duration"]),
            video_client,
            settings,
            work_dir / source["source_id"],
        )
        summaries.append(
            {
                "source_id": source["source_id"],
                "file_name": source["path"].name,
                "duration": source["duration"],
                "summary": summary,
            }
        )

    aggregate_raw = video_client.generate(
        prompt=VIDEO_COLLECTION_SUMMARY_PROMPT_TEMPLATE.format(
            video_duration=sum(float(item["duration"]) for item in source_videos),
            source_summaries=str(summaries),
        ),
        system_instruction="You merge multiple source-video summaries into one remix-oriented understanding.",
        response_mime_type="application/json",
    )
    merged = _ensure_whole_video_summary(extract_json_block(aggregate_raw))
    merged["source_video_count"] = len(source_videos)
    merged["source_videos"] = [
        {"source_id": item["source_id"], "file_name": item["file_name"], "duration": item["duration"]}
        for item in summaries
    ]
    _log(f"Full source understanding complete: {merged['summary']}")
    return merged


def _compact_whole_video_context(summary: dict[str, Any]) -> str:
    if not summary:
        return "No full-video summary available."
    payload = {
        "summary": summary.get("summary", ""),
        "story_overview": summary.get("story_overview", ""),
        "main_characters": summary.get("main_characters", []),
        "key_themes": summary.get("key_themes", []),
        "tone_arc": summary.get("tone_arc", ""),
        "emotional_peaks": summary.get("emotional_peaks", [])[:6],
        "mixing_guidance": summary.get("mixing_guidance", ""),
    }
    return str(payload)


def _ensure_whole_video_summary(raw: Any) -> dict[str, Any]:
    fallback = {
        "summary": "Whole video summary unavailable.",
        "story_overview": "",
        "main_characters": [],
        "key_themes": [],
        "tone_arc": "",
        "emotional_peaks": [],
        "mixing_guidance": "Emphasize source audio on strong dialogue or emotional reaction shots when appropriate.",
    }
    if not isinstance(raw, dict):
        return fallback
    return {
        "summary": str(raw.get("summary", fallback["summary"])).strip() or fallback["summary"],
        "story_overview": str(raw.get("story_overview", fallback["story_overview"])).strip(),
        "main_characters": list(raw.get("main_characters", fallback["main_characters"]))[:12] if isinstance(raw.get("main_characters"), list) else fallback["main_characters"],
        "key_themes": list(raw.get("key_themes", fallback["key_themes"]))[:12] if isinstance(raw.get("key_themes"), list) else fallback["key_themes"],
        "tone_arc": str(raw.get("tone_arc", fallback["tone_arc"])).strip(),
        "emotional_peaks": list(raw.get("emotional_peaks", fallback["emotional_peaks"]))[:8] if isinstance(raw.get("emotional_peaks"), list) else fallback["emotional_peaks"],
        "mixing_guidance": str(raw.get("mixing_guidance", fallback["mixing_guidance"])).strip() or fallback["mixing_guidance"],
    }


def _ensure_chunk_summary_payload(raw: Any, start_sec: float, end_sec: float) -> dict[str, Any]:
    fallback = {
        "chunk_summary": f"Video chunk from {start_sec:.2f}s to {end_sec:.2f}s.",
        "story_progress": "",
        "characters": [],
        "key_events": [],
        "dominant_emotion": "unknown",
        "source_audio_moments": [],
        "highlight_level": "medium",
    }
    if not isinstance(raw, dict):
        return fallback
    highlight_level = str(raw.get("highlight_level", fallback["highlight_level"])).strip().lower() or fallback["highlight_level"]
    if highlight_level not in {"low", "medium", "high"}:
        highlight_level = fallback["highlight_level"]
    return {
        "chunk_summary": str(raw.get("chunk_summary", fallback["chunk_summary"])).strip() or fallback["chunk_summary"],
        "story_progress": str(raw.get("story_progress", fallback["story_progress"])).strip(),
        "characters": list(raw.get("characters", fallback["characters"]))[:10] if isinstance(raw.get("characters"), list) else fallback["characters"],
        "key_events": list(raw.get("key_events", fallback["key_events"]))[:10] if isinstance(raw.get("key_events"), list) else fallback["key_events"],
        "dominant_emotion": str(raw.get("dominant_emotion", fallback["dominant_emotion"])).strip() or fallback["dominant_emotion"],
        "source_audio_moments": list(raw.get("source_audio_moments", fallback["source_audio_moments"]))[:8] if isinstance(raw.get("source_audio_moments"), list) else fallback["source_audio_moments"],
        "highlight_level": highlight_level,
    }


def _ensure_candidate_payload(raw: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return fallback
    emotional_intensity = raw.get("emotional_intensity", fallback["emotional_intensity"])
    try:
        emotional_intensity = int(emotional_intensity)
    except Exception:
        emotional_intensity = fallback["emotional_intensity"]
    emotional_intensity = max(1, min(5, emotional_intensity))
    suitability_score = raw.get("suitability_score", fallback["suitability_score"])
    try:
        suitability_score = int(suitability_score)
    except Exception:
        suitability_score = fallback["suitability_score"]
    suitability_score = max(1, min(5, suitability_score))
    return {
        "summary": str(raw.get("summary", fallback["summary"])).strip() or fallback["summary"],
        "subjects": list(raw.get("subjects", fallback["subjects"]))[:8] if isinstance(raw.get("subjects"), list) else fallback["subjects"],
        "actions": list(raw.get("actions", fallback["actions"]))[:8] if isinstance(raw.get("actions"), list) else fallback["actions"],
        "mood": str(raw.get("mood", fallback["mood"])).strip() or fallback["mood"],
        "visual_tags": list(raw.get("visual_tags", fallback["visual_tags"]))[:8] if isinstance(raw.get("visual_tags"), list) else fallback["visual_tags"],
        "edit_use": str(raw.get("edit_use", fallback["edit_use"])).strip() or fallback["edit_use"],
        "suitability_score": suitability_score,
        "emotional_intensity": emotional_intensity,
        "source_audio_priority": bool(raw.get("source_audio_priority", fallback["source_audio_priority"])),
        "narrative_role": str(raw.get("narrative_role", fallback["narrative_role"])).strip() or fallback["narrative_role"],
    }


def _ensure_audio_payload(raw: Any, audio_duration: float) -> dict[str, Any]:
    fallback = {
        "summary": "Single-section soundtrack.",
        "sections": [{"start_sec": 0.0, "end_sec": audio_duration, "mood": "steady", "energy": "medium", "edit_purpose": "cover the full track"}],
    }
    if not isinstance(raw, dict):
        return fallback
    sections = raw.get("sections")
    if not isinstance(sections, list) or not sections:
        return fallback
    normalized: list[dict[str, Any]] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        start_sec = float(section.get("start_sec", 0.0))
        end_sec = float(section.get("end_sec", audio_duration))
        if end_sec <= start_sec:
            continue
        normalized.append(
            {
                "start_sec": round(_clamp(start_sec, 0.0, audio_duration), 3),
                "end_sec": round(_clamp(end_sec, 0.0, audio_duration), 3),
                "mood": str(section.get("mood", "steady")).strip() or "steady",
                "energy": str(section.get("energy", "medium")).strip() or "medium",
                "edit_purpose": str(section.get("edit_purpose", "")).strip() or "support the soundtrack",
            }
        )
    if not normalized:
        return fallback
    normalized.sort(key=lambda item: item["start_sec"])
    normalized[0]["start_sec"] = 0.0
    normalized[-1]["end_sec"] = round(audio_duration, 3)
    return {"summary": str(raw.get("summary", fallback["summary"])).strip() or fallback["summary"], "sections": normalized}


def _build_slots(audio_duration: float, sections: list[dict[str, Any]], target_slot_seconds: float) -> list[dict[str, Any]]:
    slot_count = max(1, math.ceil(audio_duration / max(1.0, target_slot_seconds)))
    slot_duration = audio_duration / slot_count
    slots: list[dict[str, Any]] = []
    for index in range(slot_count):
        start_sec = round(index * slot_duration, 3)
        end_sec = round(audio_duration if index == slot_count - 1 else (index + 1) * slot_duration, 3)
        midpoint = (start_sec + end_sec) / 2
        section = next(
            (
                item
                for item in sections
                if item["start_sec"] <= midpoint <= item["end_sec"] or midpoint < item["end_sec"]
            ),
            sections[-1],
        )
        slots.append(
            {
                "slot_index": index,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration": round(end_sec - start_sec, 3),
                "mood": section["mood"],
                "energy": section["energy"],
                "edit_purpose": section["edit_purpose"],
            }
        )
    return slots


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9一-鿿]+", text.lower()) if len(token) >= 2}


def _selected_source_range(candidate: dict[str, Any], slot_duration: float) -> tuple[float, float]:
    candidate_start = float(candidate["start"])
    candidate_end = float(candidate["end"])
    candidate_duration = max(0.0, candidate_end - candidate_start)
    selected_duration = min(candidate_duration, max(0.0, float(slot_duration)))
    if selected_duration <= 0:
        return round(candidate_start, 3), round(candidate_start, 3)
    source_end = candidate_end
    source_start = max(candidate_start, source_end - selected_duration)
    return round(source_start, 3), round(source_end, 3)


def _selected_frame_range(candidate: dict[str, Any], slot_duration: float) -> tuple[int, int]:
    source_start, source_end = _selected_source_range(candidate, slot_duration)
    fps = float(candidate.get("source_video_fps", 25.0) or 25.0)
    if fps <= 0:
        fps = 25.0
    frame_start = max(0, int(math.floor(source_start * fps + 1e-9)))
    frame_end = max(frame_start, int(math.ceil(source_end * fps - 1e-9)) - 1)
    return frame_start, frame_end


def _build_used_frame_range(candidate: dict[str, Any], slot: dict[str, Any]) -> dict[str, Any]:
    source_start, source_end = _selected_source_range(candidate, slot["duration"])
    frame_start, frame_end = _selected_frame_range(candidate, slot["duration"])
    return {
        "candidate_id": candidate["candidate_id"],
        "slot_index": slot["slot_index"],
        "source_video_path": str(candidate.get("source_video_path", "")),
        "source_start": source_start,
        "source_end": source_end,
        "frame_start": frame_start,
        "frame_end": frame_end,
    }


def _candidate_conflicts_used_frames(
    candidate: dict[str, Any],
    slot: dict[str, Any],
    used_frame_ranges: list[dict[str, Any]],
) -> bool:
    source_video_path = str(candidate.get("source_video_path", ""))
    frame_start, frame_end = _selected_frame_range(candidate, slot["duration"])
    for existing in used_frame_ranges:
        if source_video_path != str(existing.get("source_video_path", "")):
            continue
        existing_start = int(existing.get("frame_start", 0))
        existing_end = int(existing.get("frame_end", -1))
        if frame_start <= existing_end and existing_start <= frame_end:
            return True
    return False


def _candidate_conflicts_temporally(
    candidate: dict[str, Any],
    selected_candidates: list[dict[str, Any]],
    min_source_separation_seconds: float,
) -> bool:
    candidate_center = (float(candidate["start"]) + float(candidate["end"])) / 2
    for existing in selected_candidates:
        if str(candidate.get("source_video_path", "")) != str(existing.get("source_video_path", "")):
            continue
        overlap = min(float(candidate["end"]), float(existing["end"])) - max(float(candidate["start"]), float(existing["start"]))
        if overlap > 0.35:
            return True
        existing_center = (float(existing["start"]) + float(existing["end"])) / 2
        if abs(candidate_center - existing_center) < min_source_separation_seconds:
            return True
    return False


def _rank_candidates_for_slot(prompt: str, slot: dict[str, Any], candidates: list[dict[str, Any]], used: set[str]) -> list[dict[str, Any]]:
    prompt_tokens = _tokenize(prompt)
    slot_tokens = _tokenize(" ".join([slot["mood"], slot["energy"], slot["edit_purpose"]]))
    ranked: list[tuple[float, dict[str, Any]]] = []
    for candidate in candidates:
        if candidate["duration"] + 0.05 < slot["duration"]:
            continue
        text_blob = " ".join(
            [
                candidate.get("summary", ""),
                candidate.get("mood", ""),
                " ".join(candidate.get("visual_tags", [])),
                " ".join(candidate.get("actions", [])),
                candidate.get("edit_use", ""),
                candidate.get("narrative_role", ""),
            ]
        ).lower()
        clip_tokens = _tokenize(text_blob)
        duration_gap = max(0.0, candidate["duration"] - slot["duration"])
        score = float(candidate.get("suitability_score", 3))
        score += float(candidate.get("emotional_intensity", 3)) * 0.15
        score += len(prompt_tokens & clip_tokens) * 1.2
        score += len(slot_tokens & clip_tokens) * 1.6
        if slot["energy"] == "high" and int(candidate.get("emotional_intensity", 3)) >= 4:
            score += 1.0
        if candidate.get("source_audio_priority") and slot["energy"] == "high":
            score += 0.8
        if candidate["candidate_id"] in used:
            score -= 2.5
        score -= duration_gap * 0.45
        if duration_gap <= 0.8:
            score += 1.2
        elif duration_gap <= 1.5:
            score += 0.5
        elif duration_gap >= max(2.5, slot["duration"] * 0.45):
            score -= 1.5
        ranked.append((score, candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in ranked]


def _pick_candidate_for_slot(
    prompt: str,
    slot: dict[str, Any],
    candidates: list[dict[str, Any]],
    used_candidates: set[str],
    selected_candidates: list[dict[str, Any]],
    used_frame_ranges: list[dict[str, Any]],
    min_source_separation_seconds: float,
    preferred_candidate_id: str | None = None,
) -> dict[str, Any]:
    ranked = _rank_candidates_for_slot(prompt, slot, candidates, used_candidates)
    if not ranked:
        ranked = sorted(candidates, key=lambda item: item["duration"], reverse=True)
    if not ranked:
        raise RuntimeError("No candidates available for slot planning")

    candidate_map = {candidate["candidate_id"]: candidate for candidate in candidates}
    preferred_candidate = candidate_map.get(preferred_candidate_id or "")
    if (
        preferred_candidate
        and preferred_candidate["duration"] + 0.05 >= slot["duration"]
        and preferred_candidate["candidate_id"] not in used_candidates
        and not _candidate_conflicts_temporally(preferred_candidate, selected_candidates, min_source_separation_seconds)
        and not _candidate_conflicts_used_frames(preferred_candidate, slot, used_frame_ranges)
    ):
        return preferred_candidate

    duration_slack_limit = max(0.8, min(1.8, slot["duration"] * 0.35))
    for candidate in ranked:
        if candidate["candidate_id"] in used_candidates:
            continue
        if _candidate_conflicts_temporally(candidate, selected_candidates, min_source_separation_seconds):
            continue
        if _candidate_conflicts_used_frames(candidate, slot, used_frame_ranges):
            continue
        if candidate["duration"] - slot["duration"] <= duration_slack_limit:
            return candidate

    for candidate in ranked:
        if candidate["candidate_id"] in used_candidates:
            continue
        if _candidate_conflicts_temporally(candidate, selected_candidates, min_source_separation_seconds):
            continue
        if _candidate_conflicts_used_frames(candidate, slot, used_frame_ranges):
            continue
        return candidate

    if preferred_candidate and preferred_candidate["duration"] + 0.05 >= slot["duration"]:
        return preferred_candidate
    return ranked[0]


def _fallback_plan(
    prompt: str,
    slots: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    used_candidates: set[str] | None = None,
    min_source_separation_seconds: float = 12.0,
    initial_selected_candidates: list[dict[str, Any]] | None = None,
    initial_used_frame_ranges: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    used = set(used_candidates or set())
    selection: list[dict[str, Any]] = []
    selected_candidate_items: list[dict[str, Any]] = list(initial_selected_candidates or [])
    used_frame_ranges: list[dict[str, Any]] = list(initial_used_frame_ranges or [])
    for slot in slots:
        chosen = _pick_candidate_for_slot(
            prompt,
            slot,
            candidates,
            used,
            selected_candidate_items,
            used_frame_ranges,
            min_source_separation_seconds,
        )
        used.add(chosen["candidate_id"])
        selected_candidate_items.append(chosen)
        used_frame_ranges.append(_build_used_frame_range(chosen, slot))
        max_trim = max(0.0, chosen["duration"] - slot["duration"])
        selection.append(
            {
                "slot_index": slot["slot_index"],
                "candidate_id": chosen["candidate_id"],
                "trim_start_sec": round(max_trim, 3),
                "reason": "fallback selection",
            }
        )
    return {"selection": selection}


def _validate_plan(
    raw_plan: Any,
    prompt: str,
    slots: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    used_candidates: set[str] | None = None,
    min_source_separation_seconds: float = 12.0,
    initial_selected_candidates: list[dict[str, Any]] | None = None,
    initial_used_frame_ranges: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(raw_plan, dict) or not isinstance(raw_plan.get("selection"), list):
        return _fallback_plan(
            prompt,
            slots,
            candidates,
            used_candidates=used_candidates,
            min_source_separation_seconds=min_source_separation_seconds,
            initial_selected_candidates=initial_selected_candidates,
            initial_used_frame_ranges=initial_used_frame_ranges,
        )

    used = set(used_candidates or set())
    normalized: list[dict[str, Any]] = []
    selected_candidate_items: list[dict[str, Any]] = list(initial_selected_candidates or [])
    used_frame_ranges: list[dict[str, Any]] = list(initial_used_frame_ranges or [])
    raw_items = sorted(
        [item for item in raw_plan["selection"] if isinstance(item, dict)],
        key=lambda item: int(item.get("slot_index", 10**9)),
    )

    for slot in slots:
        raw_item = next((item for item in raw_items if int(item.get("slot_index", -1)) == slot["slot_index"]), None)
        preferred_candidate_id = str(raw_item.get("candidate_id", "")) if raw_item else ""
        candidate = _pick_candidate_for_slot(
            prompt,
            slot,
            candidates,
            used,
            selected_candidate_items,
            used_frame_ranges,
            min_source_separation_seconds,
            preferred_candidate_id=preferred_candidate_id,
        )
        candidate_id = candidate["candidate_id"]
        max_trim = max(0.0, candidate["duration"] - slot["duration"])
        trim_start = float(raw_item.get("trim_start_sec", max_trim)) if raw_item is not None else max_trim
        trim_start = round(_clamp(trim_start, 0.0, max_trim), 3)
        reason = str(raw_item.get("reason", "")).strip() if raw_item is not None else ""
        if max_trim > 0:
            trim_start = round(max_trim, 3)
        if candidate_id != preferred_candidate_id:
            trim_start = round(max_trim, 3)
            reason = "deduped selection"
        normalized.append(
            {
                "slot_index": slot["slot_index"],
                "candidate_id": candidate_id,
                "trim_start_sec": trim_start,
                "reason": reason or "selected by planner",
            }
        )
        used.add(candidate_id)
        selected_candidate_items.append(candidate)
        used_frame_ranges.append(_build_used_frame_range(candidate, slot))

    return {"selection": normalized}


def _slot_peak_score(slot: dict[str, Any]) -> int:
    text = " ".join(
        [
            str(slot.get("mood", "")),
            str(slot.get("energy", "")),
            str(slot.get("edit_purpose", "")),
        ]
    ).lower()
    score = 0
    if slot.get("energy") == "high":
        score += 3
    elif slot.get("energy") == "medium":
        score += 1

    if any(term in text for term in ("chorus", "hook", "drop", "climax", "peak", "高潮", "副歌", "爆发", "爆点", "卡点")):
        score += 3
    if any(term in text for term in ("dialogue", "reveal", "conflict", "shout", "cry", "情绪", "对话", "冲突", "反转", "呐喊", "哭", "告白")):
        score += 2
    return score


def _select_highlight_indices(timeline: list[dict[str, Any]], settings: Settings) -> set[int]:
    eligible: list[tuple[float, int]] = []
    for index, item in enumerate(timeline):
        peak_score = int(item.get("slot_peak_score", 0))
        emotional_intensity = int(item.get("emotional_intensity", 3))
        source_audio_priority = bool(item.get("source_audio_priority"))

        has_strong_native_reason = source_audio_priority or (emotional_intensity >= 5 and peak_score >= 6)
        if not has_strong_native_reason:
            continue
        if peak_score < 4:
            continue

        score = peak_score * 2.2 + emotional_intensity * 0.7 + (3.0 if source_audio_priority else 0.0)
        eligible.append((score, index))

    if not eligible:
        return set()

    desired = math.ceil(len(timeline) * settings.max_highlight_audio_ratio)
    desired = max(settings.min_highlight_audio_segments, desired)
    desired = min(desired, settings.max_highlight_audio_segments, len(eligible))
    if desired <= 0:
        return set()

    eligible.sort(key=lambda item: item[0], reverse=True)
    cutoff = max(13.0, eligible[0][0] - 3.0)
    selected: list[int] = []
    for score, index in eligible:
        if len(selected) >= desired:
            break
        if score < cutoff:
            continue
        selected.append(index)
    return set(selected)


def _build_timeline(plan: dict[str, Any], slots: list[dict[str, Any]], candidates: list[dict[str, Any]], settings: Settings) -> list[dict[str, Any]]:
    candidate_map = {candidate["candidate_id"]: candidate for candidate in candidates}
    provisional: list[dict[str, Any]] = []
    for item in sorted(plan["selection"], key=lambda row: row["slot_index"]):
        slot = slots[item["slot_index"]]
        candidate = candidate_map[item["candidate_id"]]
        source_start, source_end = _selected_source_range(candidate, slot["duration"])
        source_frame_start, source_frame_end = _selected_frame_range(candidate, slot["duration"])
        provisional.append(
            {
                "slot_index": slot["slot_index"],
                "candidate_id": candidate["candidate_id"],
                "source_video_path": candidate["source_video_path"],
                "source_video_name": candidate.get("source_video_name", Path(str(candidate["source_video_path"])).name),
                "slot_start": slot["start_sec"],
                "slot_end": slot["end_sec"],
                "duration": slot["duration"],
                "source_start": source_start,
                "source_end": source_end,
                "source_frame_start": source_frame_start,
                "source_frame_end": source_frame_end,
                "source_frame_label": f"{candidate['candidate_id']}[{source_frame_start}-{source_frame_end}]",
                "reason": item["reason"],
                "slot_energy": slot["energy"],
                "slot_mood": slot["mood"],
                "slot_peak_score": _slot_peak_score(slot),
                "emotional_intensity": int(candidate.get("emotional_intensity", 3)),
                "source_audio_priority": bool(candidate.get("source_audio_priority", False)),
            }
        )

    highlight_indices = _select_highlight_indices(provisional, settings)
    timeline: list[dict[str, Any]] = []
    for index, item in enumerate(provisional):
        highlight_source_audio = index in highlight_indices
        timeline.append(
            {
                **item,
                "source_audio_level": settings.source_audio_highlight_level if highlight_source_audio else settings.source_audio_normal_level,
                "music_level": settings.music_normal_level,
                "audio_mode": "highlight_native_audio" if highlight_source_audio else "music_driven",
            }
        )
    return timeline


def _auto_tune_settings(settings: Settings, video_duration: float, audio_duration: float) -> None:
    if not settings.adaptive_tuning:
        _log("Auto tuning disabled, using fixed parameters")
        return

    duration = max(video_duration, audio_duration)
    if duration <= 5 * 60:
        profile = {
            "name": "short",
            "scene_threshold": 0.32,
            "sample_frame_count": 4,
            "max_candidates": 48,
            "max_clip_seconds": 8.0,
            "target_slot_seconds": 3.5,
            "planner_batch_size": 12,
            "planner_candidate_pool": 14,
            "video_analysis_workers": 80,
        }
    elif duration <= 15 * 60:
        profile = {
            "name": "medium",
            "scene_threshold": 0.38,
            "sample_frame_count": 3,
            "max_candidates": 80,
            "max_clip_seconds": 10.0,
            "target_slot_seconds": 4.5,
            "planner_batch_size": 10,
            "planner_candidate_pool": 12,
            "video_analysis_workers": 80,
        }
    elif duration <= 30 * 60:
        profile = {
            "name": "long",
            "scene_threshold": 0.42,
            "sample_frame_count": 3,
            "max_candidates": 120,
            "max_clip_seconds": 12.0,
            "target_slot_seconds": 6.0,
            "planner_batch_size": 8,
            "planner_candidate_pool": 10,
            "video_analysis_workers": 80,
        }
    else:
        profile = {
            "name": "xlong",
            "scene_threshold": 0.48,
            "sample_frame_count": 2,
            "max_candidates": 140,
            "max_clip_seconds": 15.0,
            "target_slot_seconds": 7.5,
            "planner_batch_size": 6,
            "planner_candidate_pool": 8,
            "video_analysis_workers": 80,
        }

    env_worker_cap_raw = os.getenv("VIDEO_ANALYSIS_WORKERS_CAP", "").strip()
    if env_worker_cap_raw:
        try:
            env_worker_cap = max(1, int(env_worker_cap_raw))
            profile["video_analysis_workers"] = min(profile["video_analysis_workers"], env_worker_cap)
        except ValueError:
            pass

    settings.scene_threshold = profile["scene_threshold"]
    settings.sample_frame_count = profile["sample_frame_count"]
    settings.max_candidates = profile["max_candidates"]
    settings.max_clip_seconds = profile["max_clip_seconds"]
    settings.target_slot_seconds = profile["target_slot_seconds"]
    settings.planner_batch_size = profile["planner_batch_size"]
    settings.planner_candidate_pool = profile["planner_candidate_pool"]
    settings.video_analysis_workers = profile["video_analysis_workers"]

    _log(
        "Auto tuning profile: "
        f"{profile['name']} | scene_threshold={settings.scene_threshold} | "
        f"sample_frame_count={settings.sample_frame_count} | "
        f"max_candidates={settings.max_candidates} | "
        f"max_clip_seconds={settings.max_clip_seconds} | "
        f"target_slot_seconds={settings.target_slot_seconds} | "
        f"planner_batch_size={settings.planner_batch_size} | "
        f"video_analysis_workers={settings.video_analysis_workers}"
    )


def _summarize_whole_video_direct(video_path: str, video_duration: float, video_client) -> dict[str, Any]:
    raw = video_client.generate_from_file(
        prompt=WHOLE_VIDEO_SUMMARY_PROMPT_TEMPLATE.format(video_duration=video_duration),
        file_path=video_path,
        system_instruction="You summarize the full source video to help downstream automatic remix editing.",
    )
    return _ensure_whole_video_summary(extract_json_block(raw))


def _summarize_whole_video_chunked(
    video_path: str,
    video_duration: float,
    video_client,
    settings: Settings,
    work_dir: Path,
) -> dict[str, Any]:
    chunk_seconds = max(30, int(settings.whole_video_summary_chunk_seconds))
    chunk_count = max(1, math.ceil(video_duration / chunk_seconds))
    configured_workers, workers = _resolve_parallelism(settings.whole_video_summary_workers, chunk_count)
    progress_step = max(1, min(5, chunk_count // 12 if chunk_count > 12 else chunk_count))
    chunks_dir = work_dir / "whole_video_chunks"
    chunk_json_path = work_dir / "whole_video_chunk_summaries.json"

    _log(f"Long video detected, analyzing in {chunk_seconds}s chunks ({chunk_count} chunks)")
    chunk_items: list[dict[str, Any]] = []
    for index in range(chunk_count):
        start_sec = float(index * chunk_seconds)
        end_sec = min(video_duration, start_sec + chunk_seconds)
        duration = round(end_sec - start_sec, 3)
        chunk_path = chunks_dir / f"chunk_{index:03d}.mp4"
        extract_summary_video_chunk(video_path, start_sec, duration, chunk_path)
        chunk_items.append(
            {
                "chunk_index": index,
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "duration": duration,
                "chunk_path": str(chunk_path),
            }
        )

    _log(f"Whole-video analysis parallelism: {workers} (limit {configured_workers})")

    def _process_chunk(item: dict[str, Any]) -> dict[str, Any]:
        raw = video_client.generate_from_file(
            prompt=WHOLE_VIDEO_CHUNK_PROMPT_TEMPLATE.format(
                start_sec=item["start_sec"],
                end_sec=item["end_sec"],
                duration=item["duration"],
            ),
            file_path=item["chunk_path"],
            mime_type="video/mp4",
            system_instruction="You summarize one chunk of a longer source video to support downstream auto-editing.",
            inline_limit_bytes=settings.whole_video_summary_inline_limit_bytes,
        )
        payload = _ensure_chunk_summary_payload(
            extract_json_block(raw),
            start_sec=item["start_sec"],
            end_sec=item["end_sec"],
        )
        return {**item, **payload}

    completed = 0
    success_items: list[dict[str, Any]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_process_chunk, item): item for item in chunk_items}
        for future in as_completed(future_map):
            item = future_map[future]
            try:
                success_items.append(future.result())
            except Exception as exc:
                errors.append(f"chunk_{item['chunk_index']:03d}: {exc}")
            completed += 1
            if completed == 1 or completed == chunk_count or completed % progress_step == 0:
                _log(f"  Whole-video chunk progress: {completed}/{chunk_count}")

    success_items.sort(key=lambda item: item["chunk_index"])
    write_json(chunk_json_path, {"chunks": success_items, "errors": errors})

    if not success_items:
        raise RuntimeError("all whole-video chunk analyses failed")
    if errors:
        _log(f"Whole-video analysis had {len(errors)} failed chunks; merging remaining {len(success_items)} chunks")

    aggregate_input = [
        {
            "chunk_index": item["chunk_index"],
            "start_sec": item["start_sec"],
            "end_sec": item["end_sec"],
            "chunk_summary": item["chunk_summary"],
            "story_progress": item["story_progress"],
            "characters": item["characters"],
            "key_events": item["key_events"],
            "dominant_emotion": item["dominant_emotion"],
            "source_audio_moments": item["source_audio_moments"],
            "highlight_level": item["highlight_level"],
        }
        for item in success_items
    ]
    aggregate_raw = video_client.generate(
        prompt=WHOLE_VIDEO_AGGREGATE_PROMPT_TEMPLATE.format(
            video_duration=video_duration,
            chunk_summaries=str(aggregate_input),
        ),
        system_instruction="You merge chunk summaries into one full-video understanding for downstream remix editing.",
        response_mime_type="application/json",
    )
    summary = _ensure_whole_video_summary(extract_json_block(aggregate_raw))
    summary["chunk_count"] = chunk_count
    summary["chunk_success_count"] = len(success_items)
    summary["chunk_error_count"] = len(errors)
    return summary


def _summarize_whole_video(
    video_path: str,
    video_duration: float,
    video_client,
    settings: Settings,
    work_dir: Path,
) -> dict[str, Any]:
    _log("Starting whole-video understanding")
    if video_duration <= settings.whole_video_summary_direct_max_seconds:
        summary = _summarize_whole_video_direct(video_path, video_duration, video_client)
    else:
        summary = _summarize_whole_video_chunked(video_path, video_duration, video_client, settings, work_dir)
    _log(f"Whole-video understanding complete: {summary['summary']}")
    return summary


def _describe_candidates(
    work_dir: Path,
    candidates: list[dict[str, Any]],
    settings: Settings,
    video_client,
    whole_video_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    frames_root = work_dir / "frames"
    total = len(candidates)
    progress_step = max(1, min(5, total // 12 if total > 12 else total))
    configured_workers, max_workers = _resolve_parallelism(settings.video_analysis_workers, total)
    whole_video_context = _compact_whole_video_context(whole_video_summary)

    _log(f"Starting candidate clip analysis: {total} clips")
    _log(f"Candidate analysis parallelism: {max_workers} (limit {configured_workers})")

    def _process_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        frame_paths = extract_frame_samples(
            video_path=candidate["source_video_path"],
            start_sec=candidate["start"],
            end_sec=candidate["end"],
            frame_count=settings.sample_frame_count,
            output_dir=frames_root / candidate["candidate_id"],
        )
        raw = video_client.generate_from_images(
            prompt=VIDEO_ANALYSIS_PROMPT_TEMPLATE.format(whole_video_context=whole_video_context),
            image_paths=frame_paths,
            system_instruction="You help choose video clips for auto-editing. Be concise, factual, and aware of emotional peaks.",
        )
        fallback = {
            "summary": f"Candidate clip from {candidate['source_video_name']} {candidate['start']:.2f}s to {candidate['end']:.2f}s.",
            "subjects": [],
            "actions": [],
            "mood": "unknown",
            "visual_tags": [],
            "edit_use": "generic cutaway",
            "suitability_score": 3,
            "emotional_intensity": 3,
            "source_audio_priority": False,
            "narrative_role": "supporting shot",
        }
        payload = _ensure_candidate_payload(extract_json_block(raw), fallback)
        return {**candidate, **payload}

    described_map: dict[str, dict[str, Any]] = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_candidate, candidate): candidate for candidate in candidates}
        for future in as_completed(futures):
            candidate = futures[future]
            try:
                described_map[candidate["candidate_id"]] = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"Candidate analysis failed for {candidate['candidate_id']} "
                    f"({candidate['start']:.2f}s-{candidate['end']:.2f}s): {exc}"
                ) from exc
            completed += 1
            if completed == 1 or completed == total or completed % progress_step == 0:
                _log(f"  Candidate analysis progress: {completed}/{total}")

    described = [described_map[candidate["candidate_id"]] for candidate in candidates]
    _log("Candidate clip analysis complete")
    return described


def _analyze_audio(audio_path: str, work_dir: Path, audio_duration: float, audio_client) -> dict[str, Any]:
    _log(f"Starting audio structure analysis, duration {audio_duration:.2f}s")
    analysis_audio = transcode_audio_for_analysis(audio_path, work_dir / "analysis_audio.mp3")
    _log("  Audio transcoded, requesting audio model")
    raw = audio_client.generate_from_file(
        prompt=AUDIO_ANALYSIS_PROMPT_TEMPLATE.format(audio_duration=audio_duration),
        file_path=analysis_audio,
        mime_type="audio/mpeg",
        system_instruction="You convert music into editing structure. Use concise section-level labels.",
    )
    result = _ensure_audio_payload(extract_json_block(raw), audio_duration)
    _log(f"Audio analysis complete with {len(result['sections'])} sections")
    return result


def _build_candidate_pool(
    prompt: str,
    batch_slots: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    settings: Settings,
    used_candidates: set[str],
    selected_candidates: list[dict[str, Any]] | None = None,
    used_frame_ranges: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    score_map: dict[str, float] = {}
    candidate_map = {candidate["candidate_id"]: candidate for candidate in candidates}
    selected_candidates = list(selected_candidates or [])
    used_frame_ranges = list(used_frame_ranges or [])

    for slot in batch_slots:
        ranked = _rank_candidates_for_slot(prompt, slot, candidates, used_candidates)
        picked = 0
        for candidate in ranked:
            if _candidate_conflicts_temporally(candidate, selected_candidates, settings.min_source_separation_seconds):
                continue
            if _candidate_conflicts_used_frames(candidate, slot, used_frame_ranges):
                continue
            score_map[candidate["candidate_id"]] = score_map.get(candidate["candidate_id"], 0.0) + (4 - min(picked, 3))
            picked += 1
            if picked >= 4:
                break

    ordered_ids = sorted(score_map, key=lambda cid: score_map[cid], reverse=True)
    if not ordered_ids:
        ordered_ids = [candidate["candidate_id"] for candidate in sorted(candidates, key=lambda item: item["duration"], reverse=True)]
    filtered_ids: list[str] = []
    for cid in ordered_ids:
        candidate = candidate_map.get(cid)
        if not candidate:
            continue
        if any(
            not _candidate_conflicts_temporally(candidate, selected_candidates, settings.min_source_separation_seconds)
            and not _candidate_conflicts_used_frames(candidate, slot, used_frame_ranges)
            for slot in batch_slots
        ):
            filtered_ids.append(cid)
    if not filtered_ids:
        filtered_ids = ordered_ids
    limited_ids = filtered_ids[: settings.planner_candidate_pool]
    return [candidate_map[cid] for cid in limited_ids if cid in candidate_map]


def _compact_slots(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "slot_index": slot["slot_index"],
            "duration": slot["duration"],
            "mood": slot["mood"],
            "energy": slot["energy"],
            "edit_purpose": slot["edit_purpose"],
        }
        for slot in slots
    ]


def _source_order_key(candidate: dict[str, Any]) -> tuple[int, float, float, str]:
    return (
        int(candidate.get("source_video_index", 0)),
        float(candidate.get("start", 0.0)),
        float(candidate.get("end", 0.0)),
        str(candidate.get("candidate_id", "")),
    )


def _compact_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": candidate["candidate_id"],
            "duration": candidate["duration"],
            "summary": candidate["summary"],
            "mood": candidate["mood"],
            "visual_tags": candidate["visual_tags"][:4],
            "edit_use": candidate["edit_use"],
            "emotional_intensity": candidate.get("emotional_intensity", 3),
            "source_audio_priority": candidate.get("source_audio_priority", False),
            "narrative_role": candidate.get("narrative_role", ""),
        }
        for candidate in candidates
    ]


def _plan_edit(
    prompt: str,
    slots: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    audio_analysis: dict[str, Any],
    whole_video_summary: dict[str, Any],
    agent_client,
    settings: Settings,
) -> dict[str, Any]:
    _log(f"Starting edit planning with {len(slots)} slots and {len(candidates)} candidates")
    batch_size = max(1, settings.planner_batch_size)
    used_candidates: set[str] = set()
    selected_candidate_items: list[dict[str, Any]] = []
    used_frame_ranges: list[dict[str, Any]] = []
    candidate_lookup = {candidate["candidate_id"]: candidate for candidate in candidates}
    slot_lookup = {slot["slot_index"]: slot for slot in slots}
    all_selection: list[dict[str, Any]] = []

    for batch_start in range(0, len(slots), batch_size):
        batch_slots = slots[batch_start:batch_start + batch_size]
        batch_no = batch_start // batch_size + 1
        batch_total = math.ceil(len(slots) / batch_size)
        candidate_pool = _build_candidate_pool(
            prompt,
            batch_slots,
            candidates,
            settings,
            used_candidates,
            selected_candidate_items,
            used_frame_ranges,
        )
        _log(
            f"Planning batch {batch_no}/{batch_total}: "
            f"slots {batch_slots[0]['slot_index']}-{batch_slots[-1]['slot_index']} | "
            f"candidate pool {len(candidate_pool)}"
        )

        prompt_text = f"""
User prompt:
{prompt}

Whole video understanding:
{_compact_whole_video_context(whole_video_summary)}

Audio summary:
{audio_analysis['summary']}

Current slot batch:
{_compact_slots(batch_slots)}

Candidate pool:
{_compact_candidates(candidate_pool)}

Already used candidate ids:
{sorted(used_candidates)}

Task:
- choose one candidate clip for each slot in the current batch
- candidate duration must be greater than or equal to slot duration
- prefer diverse clips and avoid obvious repetition
- choose clips that fit both the user prompt and the slot mood
- only use candidate ids from the candidate pool above
- prefer source_audio_priority=true clips for emotionally explosive, dialogue-heavy, or dramatic reveal moments when appropriate
- trim_start_sec means how many seconds to skip from the chosen candidate's own start

Return strict JSON:
{{
  "selection": [
    {{
      "slot_index": {batch_slots[0]['slot_index']},
      "candidate_id": "{candidate_pool[0]['candidate_id'] if candidate_pool else 'clip_000'}",
      "trim_start_sec": 0.0,
      "reason": "short reason"
    }}
  ]
}}
""".strip()

        try:
            raw = agent_client.chat(
                messages=[
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=0.2,
                max_tokens=1800,
            )
            batch_plan = _validate_plan(
                extract_json_block(raw),
                prompt,
                batch_slots,
                candidate_pool,
                used_candidates=used_candidates,
                min_source_separation_seconds=settings.min_source_separation_seconds,
                initial_selected_candidates=selected_candidate_items,
                initial_used_frame_ranges=used_frame_ranges,
            )
        except Exception as exc:
            _log(f"  Planning batch {batch_no} failed, falling back to heuristic selection: {exc}")
            batch_plan = _fallback_plan(
                prompt,
                batch_slots,
                candidate_pool or candidates,
                used_candidates=used_candidates,
                min_source_separation_seconds=settings.min_source_separation_seconds,
                initial_selected_candidates=selected_candidate_items,
                initial_used_frame_ranges=used_frame_ranges,
            )

        all_selection.extend(batch_plan["selection"])
        for item in batch_plan["selection"]:
            used_candidates.add(item["candidate_id"])
            candidate = candidate_lookup.get(item["candidate_id"])
            slot = slot_lookup.get(item["slot_index"])
            if candidate:
                selected_candidate_items.append(candidate)
                if slot:
                    used_frame_ranges.append(_build_used_frame_range(candidate, slot))

    plan = {"selection": all_selection}
    _log(f"Edit planning complete, selected {len(plan['selection'])} clips")
    return plan


def run_pipeline(


    video_path: str,
    audio_path: str,
    prompt: str,
    output_dir: str | None = None,
    env_path: str | None = None,
) -> dict[str, Any]:
    total_start = time.time()
    path_base_dir = Path.cwd().resolve()
    resolved_video_input = Path(video_path).expanduser().resolve()
    source_video_paths = _resolve_source_videos(resolved_video_input)

    _log("Starting automatic editing")
    _log(f"Video input: {resolved_video_input}")
    if len(source_video_paths) == 1:
        _log(f"Single-video mode: {source_video_paths[0].name}")
    else:
        _log(f"Folder mode: {len(source_video_paths)} videos")
    _log(f"Audio input: {audio_path}")
    _log(f"Prompt: {prompt}")

    settings = load_settings(env_path=env_path)
    job_dir = Path(output_dir).expanduser().resolve() if output_dir else settings.output_root / _slugify(prompt)
    work_dir = job_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Output directory: {job_dir}")

    video_client, audio_client, agent_client = build_clients(settings)
    _log("Model clients initialized")

    _log("Reading media durations")
    audio_duration = ffprobe_duration(audio_path)
    source_videos: list[dict[str, Any]] = []
    total_source_duration = 0.0
    for index, source_path in enumerate(source_video_paths):
        duration = ffprobe_duration(source_path)
        total_source_duration += duration
        source_videos.append(
            {
                "source_id": f"src_{index:02d}",
                "index": index,
                "path": source_path,
                "duration": duration,
                "fps": ffprobe_video_fps(source_path),
            }
        )
    _log(f"Total source duration: {total_source_duration:.2f}s | Audio duration: {audio_duration:.2f}s")

    _auto_tune_settings(settings, total_source_duration, audio_duration)

    whole_video_summary: dict[str, Any] = {}
    if settings.whole_video_summary_enabled:
        stage_start = time.time()
        try:
            whole_video_summary = _summarize_video_collection(source_videos, video_client, settings, work_dir / "whole_video")
            write_json(job_dir / "whole_video_summary.json", whole_video_summary)
            _log(f"Whole-video understanding stage complete in {time.time() - stage_start:.1f}s")
        except Exception as exc:
            _log(f"Whole-video understanding failed, continuing pipeline: {exc}")
            whole_video_summary = _ensure_whole_video_summary(None)
            write_json(job_dir / "whole_video_summary.json", whole_video_summary)

    stage_start = time.time()
    _log("Starting scene-change detection")
    raw_candidates: list[dict[str, Any]] = []
    total_cut_points = 0
    for source in source_videos:
        source_path = source["path"]
        cut_points = detect_scene_changes(source_path, threshold=settings.scene_threshold)
        total_cut_points += len(cut_points)
        source_candidates = build_candidate_segments(
            total_duration=source["duration"],
            cut_points=cut_points,
            min_clip_seconds=settings.min_clip_seconds,
            max_clip_seconds=settings.max_clip_seconds,
        )
        for local_index, candidate in enumerate(source_candidates):
            candidate["candidate_id"] = f"{source['source_id']}_clip_{local_index:03d}"
            candidate["source_video_path"] = str(source_path)
            candidate["source_video_name"] = source_path.name
            candidate["source_video_id"] = source["source_id"]
            candidate["source_video_index"] = source["index"]
            candidate["source_video_fps"] = source["fps"]
        raw_candidates.extend(source_candidates)
        _log(f"  {source_path.name}: {len(cut_points)} cut points, {len(source_candidates)} candidates")
    _log(f"Scene detection complete: {total_cut_points} cut points, {time.time() - stage_start:.1f}s")
    _log(f"Initial candidate count: {len(raw_candidates)}")

    candidates = cap_candidates(raw_candidates, settings.max_candidates)
    if len(candidates) < len(raw_candidates):
        _log(f"Too many candidates, downsampled from {len(raw_candidates)} to {len(candidates)}")
    else:
        _log(f"Candidate count: {len(candidates)}")

    stage_start = time.time()
    candidates = _describe_candidates(work_dir, candidates, settings, video_client, whole_video_summary)
    _log(f"Candidate analysis complete in {time.time() - stage_start:.1f}s")

    stage_start = time.time()
    audio_analysis = _analyze_audio(audio_path, work_dir, audio_duration, audio_client)
    _log(f"Audio structure analysis complete in {time.time() - stage_start:.1f}s")

    slots = _build_slots(audio_duration, audio_analysis["sections"], settings.target_slot_seconds)
    _log(f"Built {len(slots)} timeline slots")

    stage_start = time.time()
    plan = _plan_edit(prompt, slots, candidates, audio_analysis, whole_video_summary, agent_client, settings)
    _log(f"Edit planning complete in {time.time() - stage_start:.1f}s")

    timeline = _build_timeline(plan, slots, candidates, settings)
    highlight_count = sum(1 for item in timeline if item["audio_mode"] == "highlight_native_audio")
    _log(f"Final timeline has {len(timeline)} clips, with {highlight_count} native-audio highlights")

    stage_start = time.time()
    _log("Starting final render")
    final_video = render_timeline(
        audio_path=audio_path,
        timeline=timeline,
        output_path=job_dir / "final_edit.mp4",
        work_dir=work_dir,
        clip_audio_fade_seconds=settings.clip_audio_fade_seconds,
        transition_duration_seconds=settings.transition_duration_seconds,
    )
    _log(f"Rendering complete in {time.time() - stage_start:.1f}s")

    candidates_for_disk = _with_relative_source_paths(candidates, path_base_dir)
    timeline_for_disk = _with_relative_source_paths(timeline, path_base_dir)
    write_json(job_dir / "candidate_clips.json", candidates_for_disk)
    write_json(job_dir / "audio_analysis.json", audio_analysis)
    write_json(job_dir / "slots.json", slots)
    write_json(job_dir / "edit_plan.json", plan)
    write_json(job_dir / "timeline.json", timeline_for_disk)
    summary = {
        "video_path": _to_relative_path(resolved_video_input, path_base_dir),
        "audio_path": _to_relative_path(Path(audio_path), path_base_dir),
        "prompt": prompt,
        "final_video": _to_relative_path(final_video, path_base_dir),
        "job_dir": _to_relative_path(job_dir, path_base_dir),
        "candidate_count": len(candidates),
        "slot_count": len(slots),
        "highlight_native_audio_segments": highlight_count,
        "whole_video_chunked": any(source["duration"] > settings.whole_video_summary_direct_max_seconds for source in source_videos),
        "source_video_count": len(source_videos),
        "source_videos": [_to_relative_path(source["path"], path_base_dir) for source in source_videos],
    }
    write_json(job_dir / "result.json", summary)
    _log(f"Automatic editing complete, total time {time.time() - total_start:.1f}s")
    _log(f"Final output: {final_video}")
    return summary
