---
name: autocut-video-editing
description: Use the bundled autocut workflow to run, debug, or modify an AI-powered video editing pipeline. Trigger this skill when Codex needs to turn source video, music, and a prompt into a final cut, troubleshoot the bundled pipeline, adapt the workflow for reuse, or explain how the included autocut project is organized.
---

# Autocut Video Editing

Use the bundled project in `autocut` as the source of truth.

## Workflow

1. Read `references/project-map.md` only when you need file-level navigation.
2. Check `autocut/.env` before any real run.
   If it does not exist, copy `autocut/.env.example` to `autocut/.env`.
3. Treat API keys as a blocking prerequisite.
   Ask the user for `GEMINI_API_KEY` and `OPENAI_API_KEY` before running the pipeline.
   Treat model names in `autocut/.env.example` as fixed defaults unless the user explicitly asks to change them.
4. Confirm the runtime tools exist before making changes or attempting a run.
   Check `python --version`, `ffmpeg -version`, and `ffprobe -version`.
5. Use `autocut/run.py` as the public entrypoint.
   Run from `autocut` with either:
   `python run.py --video-path <video> --audio-path <audio> --prompt "<prompt>"`
   or import `run()` / `run_job()` from `run.py`.
6. Inspect outputs under `autocut/output/<job_name>/`.
   Expect `final_edit.mp4` plus intermediate JSON artifacts for the same job.

## Edit Targets

- `autocut/run.py`: CLI and Python entrypoint.
- `autocut/pipeline.py`: end-to-end orchestration.
- `autocut/config.py`: `.env` loading and validation.
- `autocut/media.py`: `ffmpeg` / `ffprobe` helpers and rendering utilities.
- `autocut/llm_clients.py`: Gemini and OpenAI client wiring.

## Inputs

- `video_path` may point to one video file or a directory of source videos.
- Supported source suffixes are defined in `autocut/pipeline.py`.
- `audio_path` should point to one soundtrack file.
- `prompt` should describe pacing, tone, and intended edit style.
