from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

from app.config import AppConfig, DEFAULT_CONFIG_PATH

from .cli import add_batch_arguments, build_batch_request
from .engine import TranscriptBatchProcessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m transcript_tool",
        description="Batch analysis and rewrite helper for transcripts powered by LM Studio.",
    )
    add_batch_arguments(parser)
    return parser


def load_config(path_override: Path | None) -> AppConfig:
    return AppConfig(path_override or DEFAULT_CONFIG_PATH)


def launch_gui(config: AppConfig) -> int:
    from .gui import launch_window

    return launch_window(config)


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(args, "gui", False):
        config = load_config(getattr(args, "config", None))
        return launch_gui(config)
    config = load_config(getattr(args, "config", None))
    try:
        request = build_batch_request(args, config)
    except ValueError as exc:
        parser.error(str(exc))
    processor = TranscriptBatchProcessor(request, log_callback=_console_log)
    report = processor.run()
    summary = report.to_dict()["summary"]
    print(
        f"Processed {summary['processed']} file(s), failed {summary['failed']}. "
        f"Reports saved to {request.output_dir}.",
    )
    print(f"Detailed log: {processor.log_path}")
    return 0


def _console_log(level: str, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if level.lower() == "success":
        prefix = "[OK]"
    elif level.lower() == "warning":
        prefix = "[WARN]"
    elif level.lower() == "error":
        prefix = "[ERR]"
    else:
        prefix = "[INFO]"
    print(f"{prefix} [{timestamp}] {message}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
