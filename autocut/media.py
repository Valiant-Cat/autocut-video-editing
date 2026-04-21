from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path


TRANSITION_TARGET_FPS = 25.0
TRANSITION_LANDSCAPE_SIZE = (1280, 720)
TRANSITION_PORTRAIT_SIZE = (720, 1280)


def run_cmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"{' '.join(args)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def ffprobe_duration(media_path: str | Path) -> float:
    result = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ]
    )
    return float(result.stdout.strip())


def ffprobe_video_fps(media_path: str | Path) -> float:
    result = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ],
        check=False,
    )
    raw = result.stdout.strip()
    if not raw:
        return 25.0
    try:
        if "/" in raw:
            numerator, denominator = raw.split("/", 1)
            fps = float(numerator) / max(float(denominator), 1.0)
        else:
            fps = float(raw)
    except Exception:
        return 25.0
    if not math.isfinite(fps) or fps <= 0:
        return 25.0
    return fps


def ffprobe_video_size(media_path: str | Path) -> tuple[int, int]:
    result = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(media_path),
        ],
        check=False,
    )
    raw = result.stdout.strip()
    if not raw or "x" not in raw:
        return TRANSITION_LANDSCAPE_SIZE
    try:
        width_raw, height_raw = raw.split("x", 1)
        width = int(width_raw)
        height = int(height_raw)
    except Exception:
        return TRANSITION_LANDSCAPE_SIZE
    if width <= 0 or height <= 0:
        return TRANSITION_LANDSCAPE_SIZE
    return width, height


def ffprobe_has_audio(media_path: str | Path) -> bool:
    result = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ],
        check=False,
    )
    return bool(result.stdout.strip())


def detect_scene_changes(video_path: str | Path, threshold: float) -> list[float]:
    result = run_cmd(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(video_path),
            "-vf",
            f"select='gt(scene,{threshold})',showinfo",
            "-an",
            "-f",
            "null",
            "-",
        ]
    )
    matches = re.findall(r"pts_time:([0-9.]+)", result.stderr)
    cuts = sorted({round(float(match), 3) for match in matches if float(match) > 0})
    return cuts


def build_candidate_segments(
    total_duration: float,
    cut_points: list[float],
    min_clip_seconds: float,
    max_clip_seconds: float,
) -> list[dict]:
    boundaries = [0.0]
    boundaries.extend(point for point in sorted(cut_points) if 0 < point < total_duration)
    boundaries.append(total_duration)

    raw_segments: list[list[float]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start > 0.2:
            raw_segments.append([start, end])

    merged: list[list[float]] = []
    for start, end in raw_segments:
        if not merged:
            merged.append([start, end])
            continue
        if (end - start) < min_clip_seconds:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    clipped: list[dict] = []
    for start, end in merged:
        duration = end - start
        if duration <= max_clip_seconds:
            clipped.append({"start": round(start, 3), "end": round(end, 3), "duration": round(duration, 3)})
            continue
        chunk_count = max(2, math.ceil(duration / max_clip_seconds))
        chunk_duration = duration / chunk_count
        current = start
        for _ in range(chunk_count):
            next_end = min(end, current + chunk_duration)
            clipped.append(
                {
                    "start": round(current, 3),
                    "end": round(next_end, 3),
                    "duration": round(next_end - current, 3),
                }
            )
            current = next_end

    for index, clip in enumerate(clipped):
        clip["candidate_id"] = f"clip_{index:03d}"
    return clipped


def cap_candidates(candidates: list[dict], max_candidates: int) -> list[dict]:
    if len(candidates) <= max_candidates:
        return candidates
    selected: list[dict] = []
    last_index = len(candidates) - 1
    for slot in range(max_candidates):
        idx = round(slot * last_index / max(1, max_candidates - 1))
        selected.append(candidates[idx])
    deduped: list[dict] = []
    seen: set[str] = set()
    for candidate in selected:
        if candidate["candidate_id"] in seen:
            continue
        seen.add(candidate["candidate_id"])
        deduped.append(candidate)
    for index, clip in enumerate(deduped):
        clip["candidate_id"] = f"clip_{index:03d}"
    return deduped


def extract_frame_samples(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
    frame_count: int,
    output_dir: str | Path,
) -> list[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = max(0.3, end_sec - start_sec)
    frame_paths: list[str] = []
    for index in range(frame_count):
        ratio = (index + 1) / (frame_count + 1)
        timestamp = start_sec + duration * ratio
        frame_path = output_dir / f"frame_{index:02d}.jpg"
        run_cmd(
            [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(frame_path),
            ]
        )
        frame_paths.append(str(frame_path))
    return frame_paths


def transcode_audio_for_analysis(audio_path: str | Path, output_path: str | Path) -> str:
    run_cmd(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "32k",
            str(output_path),
        ]
    )
    return str(output_path)


def extract_summary_video_chunk(
    video_path: str | Path,
    start_sec: float,
    duration: float,
    output_path: str | Path,
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(video_path),
            "-vf",
            "scale='min(960,iw)':-2:force_original_aspect_ratio=decrease",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-c:a",
            "aac",
            "-ac",
            "1",
            "-b:a",
            "64k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    return str(output_path)


def _fade_duration(duration: float, requested: float) -> float:
    return max(0.0, min(requested, max(0.0, duration / 2 - 0.01)))


def _render_clip_with_mix(
    video_path: str | Path,
    item: dict,
    clip_path: Path,
    has_source_audio: bool,
    fade_seconds: float,
) -> None:
    duration = float(item["duration"])
    fade = _fade_duration(duration, fade_seconds)
    source_volume = float(item.get("source_audio_level", 0.12))

    if has_source_audio and source_volume > 0:
        filter_complex = (
            f"[0:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
            f"asetpts=PTS-STARTPTS,volume={source_volume},"
            f"afade=t=in:st=0:d={fade:.3f},"
            f"afade=t=out:st={max(0.0, duration - fade):.3f}:d={fade:.3f},"
            f"aresample=async=1:first_pts=0[aout]"
        )
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-ss",
            f"{item['source_start']:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(video_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "pcm_s16le",
            "-ar",
            "48000",
            str(clip_path),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=channel_layout=stereo:sample_rate=48000",
            "-ss",
            f"{item['source_start']:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(video_path),
            "-map",
            "1:v:0",
            "-map",
            "0:a:0",
            "-shortest",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "pcm_s16le",
            "-ar",
            "48000",
            str(clip_path),
        ]
    run_cmd(cmd)


def _energy_rank(value: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(str(value).lower(), 2)


def _choose_transition(left_item: dict, right_item: dict, index: int, base_duration: float, left_duration: float, right_duration: float) -> tuple[str, str, str, float]:
    left_energy = _energy_rank(left_item.get("slot_energy", "medium"))
    right_energy = _energy_rank(right_item.get("slot_energy", "medium"))
    mood_text = " ".join(
        [
            str(left_item.get("slot_mood", "")),
            str(right_item.get("slot_mood", "")),
            str(left_item.get("reason", "")),
            str(right_item.get("reason", "")),
        ]
    ).lower()
    has_native_focus = left_item.get("audio_mode") == "highlight_native_audio" or right_item.get("audio_mode") == "highlight_native_audio"
    rising = right_energy > left_energy
    falling = right_energy < left_energy

    duration = base_duration
    if has_native_focus:
        duration *= 0.7
    elif left_energy == 3 and right_energy == 3:
        duration *= 0.75
    elif left_energy == 1 and right_energy == 1:
        duration *= 1.2

    duration = min(
        duration,
        max(0.10, left_duration / 2 - 0.05),
        max(0.10, right_duration / 2 - 0.05),
    )
    duration = max(0.10, duration)

    if has_native_focus:
        return "fade", "qsin", "qsin", duration
    if any(term in mood_text for term in ("梦", "回忆", "温柔", "抒情", "sad", "soft", "gentle", "memory")):
        return "fadegrays", "tri", "tri", duration
    if any(term in mood_text for term in ("冲突", "反转", "紧张", "dramatic", "conflict", "reveal", "tense")):
        return ("circleopen" if rising else "circleclose"), "exp", "exp", duration
    if left_energy == 3 and right_energy == 3:
        return ("smoothleft" if index % 2 == 0 else "smoothright"), "exp", "exp", duration
    if rising:
        return ("radial" if index % 2 == 0 else "fadeblack"), "qsin", "exp", duration
    if falling:
        return ("fadeblack" if index % 2 == 0 else "fadegrays"), "exp", "tri", duration
    return (("fade" if index % 2 == 0 else "fadeblack"), "tri", "qsin", duration)


def _choose_transition_canvas(clip_paths: list[Path]) -> tuple[int, int]:
    if not clip_paths:
        return TRANSITION_LANDSCAPE_SIZE
    width, height = ffprobe_video_size(clip_paths[0])
    if height > width:
        return TRANSITION_PORTRAIT_SIZE
    return TRANSITION_LANDSCAPE_SIZE


def _normalized_video_filter(index: int, target_width: int, target_height: int) -> str:
    return (
        f"[{index}:v]"
        f"fps={TRANSITION_TARGET_FPS:.3f},"
        f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
        f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"format=yuv420p,setsar=1,settb=AVTB,setpts=PTS-STARTPTS"
        f"[v{index}]"
    )


def _normalized_audio_filter(index: int) -> str:
    return (
        f"[{index}:a]"
        f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
        f"aresample=async=1:first_pts=0,asetpts=PTS-STARTPTS"
        f"[a{index}]"
    )


def _compose_with_transitions(
    clip_paths: list[Path],
    timeline: list[dict],
    output_path: str | Path,
    transition_duration_seconds: float,
) -> str:
    output_path = Path(output_path)
    if len(clip_paths) == 1 or transition_duration_seconds < 0.08:
        run_cmd(
            [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-i",
                str(clip_paths[0]),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
        return str(output_path)

    durations = [ffprobe_duration(path) for path in clip_paths]
    target_width, target_height = _choose_transition_canvas(clip_paths)
    cmd = ["ffmpeg", "-hide_banner", "-y"]
    for clip_path in clip_paths:
        cmd.extend(["-i", str(clip_path)])

    filter_parts = []
    for index in range(len(clip_paths)):
        filter_parts.append(_normalized_video_filter(index, target_width, target_height))
        filter_parts.append(_normalized_audio_filter(index))

    cumulative_duration = durations[0]
    for index in range(1, len(clip_paths)):
        transition_name, curve1, curve2, transition = _choose_transition(
            timeline[index - 1],
            timeline[index],
            index,
            transition_duration_seconds,
            durations[index - 1],
            durations[index],
        )
        offset = max(0.0, cumulative_duration - transition)
        filter_parts.append(
            f"[v{index - 1}][{index}:v]xfade=transition={transition_name}:duration={transition:.3f}:offset={offset:.3f}[v{index}]"
        )
        filter_parts.append(
            f"[a{index - 1}][{index}:a]acrossfade=d={transition:.3f}:c1={curve1}:c2={curve2}[a{index}]"
        )
        cumulative_duration += durations[index] - transition

    cmd.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            f"[v{len(clip_paths) - 1}]",
            "-map",
            f"[a{len(clip_paths) - 1}]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    run_cmd(cmd)
    return str(output_path)


def _mix_final_music(
    composed_video_path: str | Path,
    music_path: str | Path,
    output_path: str | Path,
    music_volume: float,
) -> str:
    output_path = Path(output_path)
    video_duration = ffprobe_duration(composed_video_path)
    filter_complex = (
        f"[0:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,asetpts=PTS-STARTPTS[src];"
        f"[1:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,asetpts=PTS-STARTPTS,volume={music_volume},atrim=0:{video_duration:.3f},aresample=async=1:first_pts=0[bg];"
        f"[src][bg]amix=inputs=2:normalize=0:duration=first,aresample=async=1:first_pts=0[aout]"
    )
    run_cmd(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(composed_video_path),
            "-i",
            str(music_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    return str(output_path)


def render_timeline(
    audio_path: str | Path,
    timeline: list[dict],
    output_path: str | Path,
    work_dir: str | Path,
    clip_audio_fade_seconds: float = 0.35,
    transition_duration_seconds: float = 0.25,
) -> str:
    work_dir = Path(work_dir)
    clips_dir = work_dir / "render_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_paths: list[Path] = []
    audio_cache: dict[str, bool] = {}

    for index, item in enumerate(timeline):
        source_video_path = str(item["source_video_path"])
        if source_video_path not in audio_cache:
            audio_cache[source_video_path] = ffprobe_has_audio(source_video_path)
        clip_path = clips_dir / f"clip_{index:03d}.mkv"
        _render_clip_with_mix(
            video_path=source_video_path,
            item=item,
            clip_path=clip_path,
            has_source_audio=audio_cache[source_video_path],
            fade_seconds=clip_audio_fade_seconds,
        )
        clip_paths.append(clip_path)

    composed_video = work_dir / "composed_source_audio.mkv"
    _compose_with_transitions(clip_paths, timeline, composed_video, transition_duration_seconds)
    music_level = float(timeline[0].get("music_level", 1.0)) if timeline else 1.0
    return _mix_final_music(composed_video, audio_path, output_path, music_level)


def write_json(path: str | Path, data: object) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
