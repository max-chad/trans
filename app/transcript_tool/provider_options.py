from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def parse_provider_preferences(source: object, *, allow_file_refs: bool = False) -> Dict[str, Any]:
    """
    Normalize provider routing preferences coming from CLI strings, config dicts, or files.

    `source` can be:
      * None -> returns {}
      * dict -> shallow-copied and returned
      * str  -> treated as JSON; if allow_file_refs is True the string may start with "@"
               (path to file) or refer to an existing JSON file on disk
      * Path -> file that will be read as JSON
    """
    if source is None:
        return {}
    if isinstance(source, dict):
        return dict(source)
    if isinstance(source, Path):
        text = Path(source).expanduser().read_text(encoding="utf-8")
        return _parse_provider_text(text)
    if isinstance(source, str):
        text = source.strip()
        if not text:
            return {}
        if allow_file_refs:
            if text.startswith("@"):
                text = _read_file_text(text[1:])
            else:
                candidate = Path(text).expanduser()
                if candidate.is_file() and not text.lstrip().startswith("{"):
                    text = candidate.read_text(encoding="utf-8")
        return _parse_provider_text(text)
    raise ValueError("Provider preferences must be a dict, JSON object, or path to JSON.")


def _read_file_text(path_str: str) -> str:
    path = Path(path_str).expanduser()
    if not path.is_file():
        raise ValueError(f"Provider preferences file not found: {path}")
    return path.read_text(encoding="utf-8")


def _parse_provider_text(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Provider preferences must be valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Provider preferences JSON must describe an object.")
    return dict(data)
