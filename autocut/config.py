from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = ROOT_DIR / ".env"


def load_env_file(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        values[key] = value
        os.environ.setdefault(key, value)
    return values


@dataclass
class Settings:
    root_dir: Path
    env_path: Path
    output_root: Path
    video_model: str
    video_api_key: str
    video_base_url: str
    audio_model: str
    audio_api_key: str
    audio_base_url: str
    agent_model: str
    agent_api_key: str
    agent_base_url: str
    scene_threshold: float = 0.32
    min_clip_seconds: float = 2.0
    max_clip_seconds: float = 8.0
    sample_frame_count: int = 4
    max_candidates: int = 36
    target_slot_seconds: float = 3.5
    request_timeout: int = 300
    adaptive_tuning: bool = True
    planner_batch_size: int = 12
    planner_candidate_pool: int = 14
    video_analysis_workers: int = 80
    whole_video_summary_enabled: bool = True
    whole_video_summary_direct_max_seconds: int = 180
    whole_video_summary_chunk_seconds: int = 60
    whole_video_summary_workers: int = 80
    whole_video_summary_inline_limit_bytes: int = 18 * 1024 * 1024
    source_audio_normal_level: float = 0.08
    source_audio_highlight_level: float = 0.62
    music_normal_level: float = 1.0
    music_duck_level: float = 1.0
    clip_audio_fade_seconds: float = 0.35
    max_highlight_audio_ratio: float = 0.12
    min_highlight_audio_segments: int = 0
    max_highlight_audio_segments: int = 4
    transition_duration_seconds: float = 0.25
    min_source_separation_seconds: float = 12.0


def _read_setting(values: dict[str, str], key: str, default: str = "") -> str:
    return os.getenv(key, values.get(key, default))


def _read_first_setting(values: dict[str, str], *keys: str, default: str = "") -> str:
    for key in keys:
        value = _read_setting(values, key)
        if value:
            return value
    return default


def load_settings(env_path: str | Path | None = None, output_root: str | Path | None = None) -> Settings:
    resolved_env = Path(env_path).expanduser().resolve() if env_path else DEFAULT_ENV_PATH
    values = load_env_file(resolved_env)
    resolved_output = Path(output_root).expanduser().resolve() if output_root else ROOT_DIR / "output"
    settings = Settings(
        root_dir=ROOT_DIR,
        env_path=resolved_env,
        output_root=resolved_output,
        video_model=_read_setting(values, "VIDEO_MODEL"),
        video_api_key=_read_first_setting(values, "GEMINI_API_KEY", "VIDEO_API_KEY"),
        video_base_url=_read_first_setting(values, "GEMINI_BASE_URL", "VIDEO_BASE_URL"),
        audio_model=_read_setting(values, "AUDIO_MODEL"),
        audio_api_key=_read_first_setting(values, "GEMINI_API_KEY", "AUDIO_API_KEY"),
        audio_base_url=_read_first_setting(values, "GEMINI_BASE_URL", "AUDIO_BASE_URL"),
        agent_model=_read_setting(values, "AGENT_MODEL"),
        agent_api_key=_read_first_setting(values, "OPENAI_API_KEY", "AGENT_API_KEY"),
        agent_base_url=_read_first_setting(values, "OPENAI_BASE_URL", "AGENT_BASE_URL"),
    )
    validate_settings(settings)
    settings.output_root.mkdir(parents=True, exist_ok=True)
    return settings


def validate_settings(settings: Settings) -> None:
    required = {
        "VIDEO_MODEL": settings.video_model,
        "GEMINI_API_KEY": settings.video_api_key,
        "AUDIO_MODEL": settings.audio_model,
        "AGENT_MODEL": settings.agent_model,
        "OPENAI_API_KEY": settings.agent_api_key,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"Missing required settings in {settings.env_path}: {', '.join(missing)}")
