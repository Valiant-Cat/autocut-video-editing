from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline import run_pipeline


# Edit these values directly if you want to run this file without CLI arguments.
# Model settings are loaded automatically from ./.env.
RUN_CONFIG = {
    "video_path": "/abs/path/video.mp4",
    "audio_path": "/abs/path/music.mp3",
    "prompt": "Create a rhythmic edit with clear emotional progression.",
    "output_dir": "",
}


def run_job(
    video_path: str,
    audio_path: str,
    prompt: str,
    output_dir: str | None = None,
) -> dict:
    return run_pipeline(
        video_path=str(Path(video_path).expanduser().resolve()),
        audio_path=str(Path(audio_path).expanduser().resolve()),
        prompt=prompt,
        output_dir=str(Path(output_dir).expanduser().resolve()) if output_dir else None,
    )


def run(
    video_path: str,
    audio_path: str,
    prompt: str,
    output_dir: str | None = None,
) -> str:
    result = run_job(
        video_path=video_path,
        audio_path=audio_path,
        prompt=prompt,
        output_dir=output_dir,
    )
    return result["final_video"]


def _resolve_run_args(args: argparse.Namespace) -> dict:
    config = dict(RUN_CONFIG)
    for key in ("video_path", "audio_path", "prompt", "output_dir"):
        value = getattr(args, key)
        if value:
            config[key] = value

    missing = [key for key in ("video_path", "audio_path", "prompt") if not config.get(key)]
    if missing:
        raise ValueError(
            "Missing required parameters: "
            + ", ".join(missing)
            + ". Set them in RUN_CONFIG or pass them by CLI."
        )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatic video editing pipeline.")
    parser.add_argument("--video-path", help="Path to the source video or a folder of source videos.")
    parser.add_argument("--audio-path", help="Path to the audio track.")
    parser.add_argument("--prompt", help="Editing prompt.")
    parser.add_argument("--output-dir", help="Optional custom output directory.")
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the full JSON result instead of only the final video path.",
    )
    args = parser.parse_args()

    run_args = _resolve_run_args(args)
    result = run_job(**run_args)

    if args.print_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"Final video generated: {result['final_video']}", flush=True)


if __name__ == "__main__":
    main()
