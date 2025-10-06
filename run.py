"""CLI entrypoint for staged transcription pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from pipeline.scheduler import PipelineScheduler


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audio/Video transcription pipeline")
    parser.add_argument("command", choices=["stage-a", "stage-b", "stage-c", "all"], help="Stage to run")
    parser.add_argument("--inputs", required=True, help="Path to input directory")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--log-level", default=None, help="Logging level override")
    return parser


def main(argv: Any = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    config = load_config(config_path)
    log_level = args.log_level or config.get("runtime", {}).get("log_level", "INFO")
    configure_logging(log_level)
    scheduler = PipelineScheduler(config)
    inputs = Path(args.inputs)
    inputs.mkdir(parents=True, exist_ok=True)

    command = args.command
    if command == "stage-a":
        scheduler.run_stage("A", inputs)
    elif command == "stage-b":
        scheduler.run_stage("B", inputs)
    elif command == "stage-c":
        scheduler.run_stage("C", inputs)
    elif command == "all":
        scheduler.run_all(inputs)
    else:  # pragma: no cover - argparse restricts commands
        raise ValueError(f"Unknown command {command}")


if __name__ == "__main__":  # pragma: no cover
    main()
