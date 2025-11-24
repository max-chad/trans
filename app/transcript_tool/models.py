from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class ProcessingMode(str, Enum):
    """High-level operations supported by the transcript tool."""

    ANALYSIS = "analysis"
    REWRITE = "rewrite"
    DEEP_REWRITE = "deep_rewrite"
    STORY = "story"

    @classmethod
    def from_flag(cls, flag: str) -> "ProcessingMode":
        normalized = flag.strip().lower()
        for mode in cls:
            if mode.value == normalized:
                return mode
        raise ValueError(f"Unsupported processing mode: {flag}")


@dataclass
class TranscriptStats:
    line_count: int
    word_count: int
    character_count: int
    duration_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        avg_words = self.word_count / self.line_count if self.line_count else 0.0
        avg_chars = self.character_count / self.line_count if self.line_count else 0.0
        return {
            "line_count": self.line_count,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "duration_seconds": self.duration_seconds,
            "avg_words_per_line": round(avg_words, 2),
            "avg_chars_per_line": round(avg_chars, 2),
        }


@dataclass
class TranscriptDocument:
    path: Path
    kind: str  # "srt", "txt", or "telegram_json"
    lines: List[str]
    stats: TranscriptStats
    srt_entries: Optional[List[Any]] = None

    def short_name(self) -> str:
        return self.path.name


@dataclass
class BatchRequest:
    input_dir: Path
    output_dir: Path
    language_hint: str = "auto"
    recursive: bool = True
    operations: Set[ProcessingMode] = field(default_factory=lambda: {ProcessingMode.ANALYSIS})
    batch_size: int = 0
    prompt_token_limit: int = 8192
    max_files: Optional[int] = None
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    preset_id: str = "1"
    merge_story_output: bool = False
    provider_preferences: Dict[str, Any] = field(default_factory=dict)
    resume: bool = True
    prompt_packing: bool = True
    reasoning_effort: Optional[str] = None


@dataclass
class FileResult:
    document: TranscriptDocument
    stats: TranscriptStats
    analysis_path: Optional[Path] = None
    rewrite_paths: Dict[str, Path] = field(default_factory=dict)
    story_path: Optional[Path] = None
    story_meta_path: Optional[Path] = None
    summary_excerpt: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "file": str(self.document.path),
            "stats": self.stats.to_dict(),
            "analysis_path": str(self.analysis_path) if self.analysis_path else None,
            "rewrite_paths": {mode: str(path) for mode, path in self.rewrite_paths.items()},
            "story_path": str(self.story_path) if self.story_path else None,
            "story_meta_path": str(self.story_meta_path) if self.story_meta_path else None,
            "summary_excerpt": self.summary_excerpt,
        }
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass
class BatchReport:
    request: BatchRequest
    processed: List[FileResult] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def record_success(self, result: FileResult) -> None:
        self.processed.append(result)

    def record_failure(self, path: Path, reason: str) -> None:
        self.failed[str(path)] = reason

    def complete(self) -> None:
        self.finished_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        duration = None
        if self.finished_at is not None:
            duration = round(self.finished_at - self.started_at, 2)
        return {
            "request": {
                "input_dir": str(self.request.input_dir),
                "output_dir": str(self.request.output_dir),
                "language_hint": self.request.language_hint,
                "operations": sorted(mode.value for mode in self.request.operations),
                "batch_size": self.request.batch_size,
                "prompt_token_limit": self.request.prompt_token_limit,
                "max_files": self.request.max_files,
                "model": self.request.model,
                "base_url": self.request.base_url,
                "prompt_packing": self.request.prompt_packing,
                "reasoning_effort": self.request.reasoning_effort,
            },
            "processed": [item.to_dict() for item in self.processed],
            "failed": self.failed,
            "summary": {
                "total_files": len(self.processed) + len(self.failed),
                "processed": len(self.processed),
                "failed": len(self.failed),
                "duration_seconds": duration,
                "total_words": sum(item.stats.word_count for item in self.processed),
                "total_minutes": round(
                    sum((item.stats.duration_seconds or 0.0) for item in self.processed) / 60.0, 2
                ),
            },
        }
