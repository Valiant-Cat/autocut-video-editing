---
name: autocut-video-editing
description: Use the bundled autocut video-editing workflow to run, debug, package, or modify an AI-powered video editing pipeline. Trigger this skill when Codex needs to turn source video, music, and an editing prompt into a final cut, troubleshoot or extend the built-in pipeline, wrap the project as a reusable workflow asset, or explain how the included project is organized and executed.
---

# Autocut Video Editing

Use the bundled project in `project/autocut` as the source of truth.

## Follow This Workflow

1. Check `project/autocut/.env` before any real run.
   If it does not exist, copy `project/autocut/.env.example` to `project/autocut/.env`.
2. Treat environment configuration as a blocking prerequisite.
   Ask the user to provide the required API keys before running the pipeline.
   Required keys: `GEMINI_API_KEY` for video and audio analysis, and `OPENAI_API_KEY` for the planning / agent stage.
   Treat model names as fixed defaults from `project/autocut/.env.example` unless the user explicitly asks to change them.
3. Confirm the runtime tools exist before making changes or attempting a run.
   Check `python --version`, `ffmpeg -version`, and `ffprobe -version`.
4. Read `references/project-map.md` when you need to locate the right file quickly.
5. Run from `project/autocut` with either:
   `python run.py --video-path <video> --audio-path <audio> --prompt "<prompt>"`
   or import `run()` / `run_job()` from `run.py`.
6. Inspect outputs under `project/autocut/output/<job_name>/`.
   Expect `final_edit.mp4` plus intermediate JSON artifacts for the same job.

## Keep These Rules In Mind

- Treat `project/autocut/run.py` as the public entrypoint.
- Treat `project/autocut/pipeline.py` as the orchestration layer.
- Do not attempt a real pipeline run with missing API keys.
- Do not ask the user to choose models unless they explicitly want to change the defaults.
- Keep `SKILL.md` short; move structural details into `references/project-map.md`.
- Do not duplicate long project documentation inside this file.
- Do not commit generated files such as `__pycache__/`, `.pyc`, or `output/`.

## Input Expectations

- `video_path` may point to a single video file or a directory of source videos.
- Supported source suffixes are defined in `project/autocut/pipeline.py`.
- `audio_path` should point to one music or soundtrack file.
- `prompt` should describe edit intent, pacing, tone, and narrative direction.

## Resource Guide

- Read `references/project-map.md` for file ownership and navigation.
- Read `project/autocut/.env.example` for required runtime configuration.
  Use the default model values there unless the user explicitly requests different models.
- Read `project/autocut/run.py` for CLI and Python entrypoints.
- Read `project/autocut/config.py` for settings loading and validation.
- Read `project/autocut/pipeline.py` for the end-to-end editing flow.
