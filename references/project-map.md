# Project Map

Bundled project root: `project/autocut`

## File Roles

- `project/autocut/run.py`
  Public entrypoint. Exposes CLI arguments plus `run()` and `run_job()` for Python callers.
- `project/autocut/pipeline.py`
  Main orchestration flow. Handles source discovery, scene segmentation, multimodal analysis, clip planning, and final render output.
- `project/autocut/media.py`
  Media utility layer around `ffmpeg` and `ffprobe`. Use this file for probing, frame extraction, candidate generation, and timeline rendering.
- `project/autocut/llm_clients.py`
  Model client adapters and JSON extraction helpers for the video, audio, and planning stages.
- `project/autocut/config.py`
  Loads `.env`, validates required settings, and defines runtime defaults.
- `project/autocut/.env.example`
  Minimal environment template for model names and API keys.
- `project/autocut/requirements.txt`
  Python dependency list for the bundled project.

## Directory Layout

```text
project/autocut/
|-- .env.example
|-- README.md
|-- requirements.txt
|-- run.py
|-- config.py
|-- llm_clients.py
|-- media.py
`-- pipeline.py
```

## What To Edit For Common Tasks

- Change CLI behavior or default local test inputs: edit `project/autocut/run.py`.
- Change environment validation or runtime knobs: edit `project/autocut/config.py`.
- Change clip discovery, probing, extraction, or rendering behavior: edit `project/autocut/media.py`.
- Change prompt design, planning, or end-to-end workflow logic: edit `project/autocut/pipeline.py`.
- Change provider wiring or request formatting: edit `project/autocut/llm_clients.py`.
