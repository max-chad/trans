# Video Transcriber (first of all nowadays it's just a beta project, not a final version)

Python desktop application for offline video transcription, translation, and post-processing.
It ships with a PyQt6 GUI, a CLI entry point, and optional integrations with Whisper, Faster
Whisper, and LM Studio for quality improvements.

## Features
- PyQt6 interface with splash screen, task queue management, and progress reporting
- CLI wrapper around the same worker pipeline for batch automation
- Whisper / Faster-Whisper transcription with selectable models and backends
- Optional LM Studio powered correction and translation steps
- Speaker diarization (SpeechBrain) with GUI toggles for on/off and device selection, plus multi-format exports (SRT / TXT)
- Diarization now prefers GPU when available (set dropdown or `DIARIZATION_DEVICE=cuda` to force)

### Speaker diarization model
- The SpeechBrain model files live in `pretrained_models/spkrec-ecapa-voxceleb`. By default the app will download them automatically (copying from the Hugging Face cache, no symlinks).
- If you want a strict offline mode, set `SPEECHBRAIN_ALLOW_DOWNLOAD=0` and place the files manually into that folder.

## Requirements
- Python 3.11 or newer (PyPI CUDA wheels target Python 3.12)
- Windows with CUDA 12+ drivers for GPU acceleration (CPU-only mode also supported)
- FFmpeg available on `PATH` (used by `moviepy`)

## Getting Started
1. Create a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
   
## Usage
### GUI
Launch the desktop interface (default when no arguments are provided):
```powershell
python main.py
```

### CLI
Run `python main.py --help` for the full list of subcommands and options. A common call:
```powershell
python main.py transcribe "path\to\input.mp4" --format srt --format txt --language auto
```
The CLI shares configuration with the GUI. Override settings on the command line or by editing
`%USERPROFILE%\.video-transcriber\projects\<project-id>\config.json`, where `<project-id>` is a
slug + hash derived from the absolute path of the current repository so different copies do not
override each other. Set `VIDEO_TRANSCRIBER_CONFIG_DIR` if you need a custom location.

### Transcript batch mode
`python main.py batch [options]` exposes `TranscriptBatchProcessor` from `app.transcript_tool`. It discovers `.srt`, `.txt`, and Telegram JSON exports in the provided directory (recursive by default), then runs LM Studio-powered analysis, rewrite, or story passes and dumps artifacts to `transcript_tool_output/` (logs, reports, rewrites, stories, merged markdown when `--merge-story` is supplied). Pass `--mode` repeatedly to combine analysis, rewrite, and story outputs or use `--provider-options`, `--prompt-limit`, and `--reasoning-effort` to control how the batch worker calls LM Studio. The command respects the same `config.json` that powers the GUI so you can reuse your LM Studio settings and provider routing preferences.

## Windows .exe builds
- Install PyInstaller if needed: `python -m pip install pyinstaller` (run inside the repo venv).
- Build both GUI and CLI executables: `.\build_exe.ps1` (add `-OneFile` to pack everything into single binaries).
- Use `dist\transcriber-gui.exe` for double-click launch without a console window.
- Use `dist\transcriber-cli.exe` when you want console logging and CLI arguments, e.g. `dist\transcriber-cli.exe transcribe input.mp4 --format srt`.


## Tests & Diagnostics
The project includes a smoke-test helper that exercises the offline pipeline without requiring a
GUI:
```powershell
python test_offline.py
```
See `CLI_USAGE.md` for additional scripts and troubleshooting tips.

## Project Structure
- `app/` — configuration, worker orchestration, CLI bindings, and GPU helpers
- `ui/` — PyQt6 widgets, splash screen, and styling
- `main.py` — entry point that dispatches between GUI and CLI modes
- `config.example.json` — starter configuration copied to the project-scoped folder under `~/.video-transcriber/projects/<project-id>/`

