from __future__ import annotations

import argparse
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Iterable, Set

from app.config import AppConfig

from .models import BatchRequest, ProcessingMode
from .provider_options import parse_provider_preferences


def add_batch_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("-i", "--input", type=Path, required=False, help="Directory with transcripts (.srt/.txt/.json).")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("transcript_tool_output"),
        help="Destination for analysis, rewrite, and story artifacts.",
    )
    parser.add_argument("--lang", default=None, help="Language hint passed to LM Studio (defaults from config).")
    parser.add_argument(
        "-m",
        "--mode",
        action="append",
        choices=[mode.value for mode in ProcessingMode],
        help="Processing mode (can be provided multiple times). Defaults to analysis.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Lines per rewrite batch (0 = auto, default: config).",
    )
    parser.add_argument("--prompt-limit", type=int, default=None, help="Prompt token limit per request (config fallback).")
    parser.add_argument("--max-files", type=int, default=None, help="Process only the first N transcripts.")
    parser.add_argument("--base-url", default=None, help="Override LM Studio base URL.")
    parser.add_argument("--model", default=None, help="Override LM Studio model identifier.")
    parser.add_argument("--api-key", default=None, help="Optional LM Studio API key.")
    parser.add_argument("--preset-id", default=None, help="LM Studio preset identifier (default: 1).")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.json (user profile by default).")
    parser.add_argument("--gui", action="store_true", help="Launch the GUI instead of running from CLI.")
    parser.add_argument(
        "--merge-story",
        action="store_true",
        help="Concatenate all generated stories into a single markdown file (Story mode only).",
    )
    parser.add_argument(
        "--provider-options",
        dest="provider_options",
        default=None,
        help="Provider routing preferences as JSON or @path (OpenRouter).",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=None,
        help="Request a specific reasoning effort (low/medium/high) when supported.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip transcripts that already have outputs (use --no-resume to force reprocessing).",
    )
    parser.add_argument(
        "--prompt-packing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pack multiple transcripts per LM Studio call to maximize token usage.",
    )


def build_batch_request(args: argparse.Namespace, config: AppConfig) -> BatchRequest:
    input_dir = args.input or Path.cwd()
    output_dir = args.output or Path("transcript_tool_output")
    language_hint = args.lang or config.get("language") or "auto"
    modes = resolve_modes(getattr(args, "mode", None)) or {ProcessingMode.ANALYSIS}
    base_url = args.base_url or config.get("lmstudio_base_url") or ""
    model = args.model or config.get("lmstudio_model") or ""
    api_key = args.api_key or config.get("lmstudio_api_key") or ""
    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        stored_batch = config.get("lmstudio_batch_size")
        try:
            batch_size = int(stored_batch)
        except (TypeError, ValueError):
            batch_size = 0
    prompt_limit = args.prompt_limit or config.get("lmstudio_prompt_token_limit") or 8192
    preset_id = args.preset_id or str(config.get("lmstudio_preset_id") or "1")
    prompt_packing = args.prompt_packing
    if prompt_packing is None:
        stored = config.get("prompt_packing_enabled")
        prompt_packing = True if stored is None else bool(stored)
    needs_llm = any(
        mode
        in {ProcessingMode.ANALYSIS, ProcessingMode.REWRITE, ProcessingMode.DEEP_REWRITE, ProcessingMode.STORY}
        for mode in modes
    )
    if needs_llm and (not base_url or not model):
        raise ValueError("LM Studio base URL and model must be specified for the selected modes.")
    reasoning_effort = _resolve_reasoning_effort(getattr(args, "reasoning_effort", None), config)
    return BatchRequest(
        input_dir=input_dir.resolve(),
        output_dir=output_dir.resolve(),
        language_hint=language_hint,
        operations=modes,
        batch_size=int(batch_size),
        prompt_token_limit=int(prompt_limit),
        max_files=args.max_files if args.max_files and args.max_files > 0 else None,
        base_url=base_url,
        model=model,
        api_key=api_key,
        preset_id=preset_id,
        merge_story_output=bool(getattr(args, "merge_story", False)),
        provider_preferences=_resolve_provider_preferences(getattr(args, "provider_options", None), config),
        reasoning_effort=reasoning_effort,
        resume=bool(getattr(args, "resume", True)),
        prompt_packing=bool(prompt_packing),
    )


def resolve_modes(mode_flags: Iterable[str] | None) -> Set[ProcessingMode]:
    modes: Set[ProcessingMode] = set()
    if not mode_flags:
        return modes
    for flag in mode_flags:
        if not flag:
            continue
        modes.add(ProcessingMode.from_flag(flag))
    return modes


def _resolve_provider_preferences(argument_value: str | None, config: AppConfig) -> Dict[str, object]:
    if argument_value:
        try:
            return parse_provider_preferences(argument_value, allow_file_refs=True)
        except ValueError as exc:
            raise ValueError(f"Invalid provider options: {exc}") from exc
    raw = config.get("lmstudio_provider_preferences")
    if raw is None:
        raw = config.get("provider_preferences")
    if raw is not None:
        try:
            return parse_provider_preferences(raw, allow_file_refs=True)
        except ValueError as exc:
            raise ValueError(f"Invalid provider options in config: {exc}") from exc
    provider_name = config.get("lmstudio_provider_name") or config.get("provider_name")
    if provider_name is None:
        return {}
    normalized_name = str(provider_name).strip()
    return {"order": [normalized_name]} if normalized_name else {}


def _resolve_reasoning_effort(argument_value: str | None, config: AppConfig) -> str | None:
    if argument_value:
        normalized = argument_value.strip().lower()
        return normalized if normalized in {"low", "medium", "high"} else None
    stored = config.get("lmstudio_reasoning_effort")
    if stored is None:
        stored = config.get("reasoning_effort")
    if stored is None:
        return None
    normalized = str(stored).strip().lower()
    return normalized if normalized in {"low", "medium", "high"} else None
