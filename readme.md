# Video Transcriber

Python desktop application for offline video transcription, translation, and post-processing.
It ships with a PyQt6 GUI, a CLI entry point, and optional integrations with Whisper, Faster
Whisper, and LM Studio for quality improvements.

## Features
- PyQt6 interface with splash screen, task queue management, and progress reporting
- CLI wrapper around the same worker pipeline for batch automation
- Whisper / Faster-Whisper transcription with selectable models and backends
- Optional LM Studio powered correction and translation steps
- Speaker diarization hooks (PyAnnote) and multi-format exports (SRT / TXT)

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
3. Copy the example configuration to your user profile and adjust as needed:
   ```powershell
   $target = Join-Path $env:USERPROFILE ".video-transcriber\config.json"
   New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
   Copy-Item config.example.json $target
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
`%USERPROFILE%\.video-transcriber\config.json`.

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
- `config.example.json` — starter configuration copied to the user profile on first launch

## Next Steps
Pick a license (`LICENSE` file) before publishing, set up CI (e.g. GitHub Actions) if desired,
and add badges or screenshots once the repository is public.
