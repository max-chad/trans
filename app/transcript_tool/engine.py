from __future__ import annotations

import codecs
import hashlib
import json
import logging
import queue
import shutil
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import os
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar
import re
from datetime import datetime, timezone
from dataclasses import dataclass

try:  # pragma: no cover - optional for environments that have not been updated yet
    import ijson
except ImportError:  # pragma: no cover
    ijson = None

import srt

from app.constants import DEFAULT_LMSTUDIO_TOKEN_MARGIN
from app.lmstudio_client import (
    LmStudioClient,
    LmStudioError,
    LmStudioSettings,
    chunked,
)
from app.token_utils import build_token_counter, configure_token_counter, estimate_tokens

from .models import (
    BatchReport,
    BatchRequest,
    FileResult,
    ProcessingMode,
    TranscriptDocument,
    TranscriptStats,
)

LogCallback = Callable[[str, str], None]

T = TypeVar("T")

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = (
    "You are an editorial analyst. Study the transcript and return JSON with the following keys: "
    '"summary" (<= 6 sentences), "topics" (array of short bullet phrases), '
    '"tone" (single adjective), "keywords" (array of 5-12 key phrases), '
    '"action_items" (array, can be empty), and "quality_flags" (array of issues such as filler words, '
    "speaker overlap, missing punctuation, etc.). Keep responses concise and stay in the transcript language."
)

STORY_SYSTEM_PROMPT = (
    "You are a narrative historian. Reshape the provided rewrite into a factual, chronological account while "
    "respecting every verifiable detail. Always write in {language}, keep an objective third-person past-tense "
    "voice, avoid fairy-tale embellishments, and tie each paragraph to a clear timeline."
)

STORY_PASS_PROMPT = (
    "Analysis summary to honor:\n{analysis_summary}\n\n"
    "Topics that must appear (cover or reinforce them when relevant):\n{topic_bullets}\n\n"
    "Quality issues to fix or watch for:\n{issue_bullets}\n\n"
    "Action items and priorities:\n{action_bullets}\n\n"
    "Previously written history excerpt (may be empty):\n{previous}\n\n"
    "Rewrite excerpt to transform:\n{chunk}\n\n"
    "Task: extend the factual history in {language}. Preserve chronology, highlight causes and effects, "
    "reference concrete facts instead of fiction, and keep transitions tight so this passage fits naturally "
    "into the overall timeline. Respond with the updated narrative for this chunk only."
)

STORY_CONTEXT_SYSTEM_PROMPT = (
    "You are a literary analyst who extracts factual context from a story. Provide structured JSON summaries."
)

STORY_CONTEXT_PROMPT = (
    "Story text:\n{story}\n\n"
    "Return JSON with keys: "
    '"characters" (array of objects with "name", "role", "traits"), '
    '"locations" (array of objects with "name", "notes"), '
    '"synopsis" (<= 120 words summary in {language}), '
    '"mood" (single adjective), '
    '"themes" (array of short phrases).'
)

EMPTY_STORY_META = {
    "characters": [],
    "locations": [],
    "synopsis": "",
    "mood": "",
    "themes": [],
}

_TOOL_OUTPUT_FOLDERS = ("analysis", "rewrites", "reports", "stories", "logs", ".temp_batches")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_DEFAULT_MAX_CONTEXT_TOKENS = 200_000
_MODEL_CONTEXT_LIMITS = {
    "openrouter/polaris-alpha": 256_000,
    "polaris-alpha": 256_000,
}
_CONTEXT_SAFETY_MARGIN = 1024
_REWRITE_JSON_FIXED_MARGIN = 1024
_REWRITE_JSON_ITEM_OVERHEAD = 1
_MIN_PROMPT_TOKENS = 512
_MIN_COMPLETION_TOKENS = 64
_CONTEXT_LIMIT_ERROR_HINTS = (
    "maximum context length",
    "context length is",
    "context window",
    "too many tokens",
    "token limit",
    "tokens exceeds",
    "prompt is too long",
)
_CREDIT_LIMIT_HINT_RE = re.compile(r"can only afford (?:up to )?([\d,]+)", re.IGNORECASE)


def _model_context_limit(model_hint: Optional[str]) -> int:
    raw = (model_hint or "").strip().lower()
    if not raw:
        return _DEFAULT_MAX_CONTEXT_TOKENS
    candidates = [raw]
    if ":" in raw:
        prefix = raw.split(":", 1)[0]
        if prefix not in candidates:
            candidates.append(prefix)
    if "/" in raw:
        suffix = raw.rsplit("/", 1)[-1]
        if suffix not in candidates:
            candidates.append(suffix)
    for candidate in candidates:
        limit = _MODEL_CONTEXT_LIMITS.get(candidate)
        if limit:
            return limit
    return _DEFAULT_MAX_CONTEXT_TOKENS


def _safe_json_loads(text: str) -> Optional[dict]:
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.split("\n")
        if len(lines) >= 2:
            candidate = "\n".join(lines[1:-1]).strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        return data
    return None


def _is_context_limit_error(exc: Exception) -> bool:
    if not isinstance(exc, LmStudioError):
        return False
    message = str(exc).lower()
    if not message:
        return False
    return any(hint in message for hint in _CONTEXT_LIMIT_ERROR_HINTS)


def _extract_credit_limit(exc: Exception) -> Optional[int]:
    if not isinstance(exc, LmStudioError):
        return None
    message = str(exc)
    if not message:
        return None
    normalized = message.lower()
    if "402" not in normalized and "credit" not in normalized and "afford" not in normalized:
        return None
    match = _CREDIT_LIMIT_HINT_RE.search(message)
    if not match:
        return None
    digits = match.group(1).replace(",", "")
    try:
        return int(digits)
    except ValueError:
        return None


def discover_transcripts(root: Path, recursive: bool = True, ignore_output: Optional[Path] = None) -> List[Path]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    patterns = ("*.srt", "*.txt", "*.json")
    paths: List[Path] = []
    if recursive:
        for pattern in patterns:
            paths.extend(sorted(root.rglob(pattern)))
    else:
        for pattern in patterns:
            paths.extend(sorted(root.glob(pattern)))
    exclude_dirs = _derive_exclude_dirs(root, ignore_output)
    unique: List[Path] = []
    seen = set()
    for path in paths:
        if path.is_file():
            resolved = path.resolve()
            if _is_under_any(resolved, exclude_dirs):
                continue
            key = str(resolved)
            if key not in seen:
                seen.add(key)
                unique.append(path)
    return sorted(unique, key=_natural_sort_key)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def load_transcript(path: Path) -> TranscriptDocument:
    suffix = path.suffix.lower()
    if suffix == ".srt":
        raw = _read_text(path)
        entries = list(srt.parse(raw))
        lines = [entry.content.strip() for entry in entries if entry.content.strip()]
        if not lines:
            raise ValueError(f"SRT file contains no subtitle lines: {path}")
        duration = None
        if entries:
            start = entries[0].start.total_seconds() if entries[0].start else 0.0
            end = entries[-1].end.total_seconds() if entries[-1].end else start
            duration = max(0.0, end - start)
        stats = _build_stats(lines, duration)
        return TranscriptDocument(path=path, kind="srt", lines=lines, stats=stats, srt_entries=entries)
    if suffix == ".txt":
        raw = _read_text(path)
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"TXT file contains no content: {path}")
        stats = _build_stats(lines, None)
        return TranscriptDocument(path=path, kind="txt", lines=lines, stats=stats, srt_entries=None)
    if suffix == ".json":
        return _load_telegram_export(path)
    raise ValueError(f"Unsupported transcript format: {path.suffix}")


def _build_stats(lines: Sequence[str], duration_seconds: Optional[float]) -> TranscriptStats:
    words = sum(len(line.split()) for line in lines)
    chars = sum(len(line) for line in lines)
    return TranscriptStats(
        line_count=len(lines),
        word_count=words,
        character_count=chars,
        duration_seconds=duration_seconds,
    )


def _load_telegram_export(path: Path) -> TranscriptDocument:
    if ijson is None:  # pragma: no cover - dependency missing only in misconfigured envs
        raise ValueError("Telegram JSON support requires the 'ijson' package to be installed.")
    lines: List[str] = []
    total_messages = 0
    for message in _iter_telegram_messages(path):
        total_messages += 1
        formatted = _format_telegram_message(message)
        if formatted:
            lines.append(formatted)
    if total_messages == 0:
        raise ValueError(f"JSON file does not look like a Telegram export with a 'messages' array: {path}")
    if not lines:
        raise ValueError(f"Telegram export contains no readable messages: {path}")
    stats = _build_stats(lines, None)
    return TranscriptDocument(path=path, kind="telegram_json", lines=lines, stats=stats, srt_entries=None)


def _rewind_telegram_stream(handle: BinaryIO, path: Path) -> None:
    """Skip UTF-8 BOM (Telegram exports on Windows) before streaming with ijson."""
    # Read a few bytes to detect BOM signature but always restore the cursor.
    prefix = handle.read(4)
    if not prefix:
        handle.seek(0)
        return
    if prefix.startswith(codecs.BOM_UTF8):
        handle.seek(len(codecs.BOM_UTF8))
        return
    if prefix.startswith(codecs.BOM_UTF16_LE):
        raise ValueError(f"{path.name} appears to be encoded as UTF-16 LE; please export JSON as UTF-8.")
    if prefix.startswith(codecs.BOM_UTF16_BE):
        raise ValueError(f"{path.name} appears to be encoded as UTF-16 BE; please export JSON as UTF-8.")
    handle.seek(0)


def _iter_telegram_messages(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with path.open("rb") as handle:
            _rewind_telegram_stream(handle, path)
            for item in ijson.items(handle, "messages.item"):
                if isinstance(item, dict):
                    yield item
    except Exception as exc:
        raise ValueError(f"Failed to parse Telegram export {path.name}: {exc}") from exc


def _format_telegram_message(message: dict[str, Any]) -> Optional[str]:
    msg_type = str(message.get("type") or "").lower()
    if msg_type != "message":
        return None
    text = _coerce_telegram_text(message.get("text"))
    if not text:
        return None
    author = message.get("from") or message.get("author") or "Unknown"
    reply_id = message.get("reply_to_message_id")
    reply_suffix = f" (reply to #{reply_id})" if reply_id else ""
    timestamp = _format_telegram_timestamp(message)
    prefix = f"[{timestamp}] " if timestamp else ""
    return f"{prefix}{author}{reply_suffix}: {text}"


def _format_telegram_timestamp(message: dict[str, Any]) -> str:
    """Return a normalized timestamp string for Telegram messages."""
    raw_date = message.get("date")
    dt: Optional[datetime] = None
    if isinstance(raw_date, str) and raw_date.strip():
        candidate = raw_date.strip()
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            dt = None
    if dt is None:
        raw_unix = message.get("date_unixtime")
        if isinstance(raw_unix, (int, float)):
            try:
                dt = datetime.fromtimestamp(float(raw_unix), tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                dt = None
        elif isinstance(raw_unix, str) and raw_unix.isdigit():
            try:
                dt = datetime.fromtimestamp(float(raw_unix), tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                dt = None
    if dt is None:
        return ""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _coerce_telegram_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: List[str] = []
        for chunk in value:
            fragment = ""
            if isinstance(chunk, str):
                fragment = chunk
            elif isinstance(chunk, dict):
                text = chunk.get("text")
                if isinstance(text, str):
                    fragment = text
            if fragment:
                parts.append(fragment)
        return "".join(parts).strip()
    if isinstance(value, dict):
        text = value.get("text") or value.get("description")
        if isinstance(text, str):
            return text.strip()
    return ""


_QUEUE_SENTINEL = object()


@dataclass
class AnalysisChunkTask:
    chunk_id: str
    snippet: str
    language_hint: str
    chunk_index: int
    chunk_total: int
    document_name: str
    prompt_tokens: int
    completion_tokens: int
    future: Future


@dataclass
class RewriteChunkTask:
    lines: List[str]
    prompt_tokens: int
    completion_tokens: int
    future: Future


class AnalysisPackingQueue:
    def __init__(
        self,
        processor: "TranscriptBatchProcessor",
        prompt_limit: int,
        completion_limit: int,
        flush_interval: float = 0.35,
    ):
        self._processor = processor
        self._prompt_limit = prompt_limit
        self._completion_limit = completion_limit
        self._flush_interval = flush_interval
        self._queue: "queue.Queue[AnalysisChunkTask | object]" = queue.Queue()
        self._stop_requested = threading.Event()
        self._worker = threading.Thread(target=self._run, name="analysis-packer", daemon=True)
        self._worker.start()

    def submit(
        self,
        snippet: str,
        language_hint: str,
        chunk_index: int,
        chunk_total: int,
        document_name: str,
        chunk_id: Optional[str] = None,
    ) -> Future:
        future: Future = Future()
        if chunk_id:
            chunk_key = chunk_id
        else:
            safe_name = document_name.replace(" ", "_")
            chunk_key = f"{safe_name}-{chunk_index}-{uuid.uuid4().hex[:6]}"
        prompt_tokens = estimate_tokens(snippet) + 256
        completion_tokens = max(256, prompt_tokens // 2)
        task = AnalysisChunkTask(
            chunk_id=chunk_key,
            snippet=snippet,
            language_hint=language_hint,
            chunk_index=chunk_index,
            chunk_total=chunk_total,
            document_name=document_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            future=future,
        )
        self._queue.put(task)
        return future

    def close(self) -> None:
        self._stop_requested.set()
        self._queue.put(_QUEUE_SENTINEL)
        self._worker.join(timeout=2.0)

    def update_limits(self, prompt_limit: int, completion_limit: int) -> None:
        self._prompt_limit = max(_MIN_PROMPT_TOKENS, prompt_limit)
        self._completion_limit = max(_MIN_COMPLETION_TOKENS, completion_limit)

    def _run(self) -> None:
        pending: List[AnalysisChunkTask] = []
        prompt_tokens = 0
        completion_tokens = 0
        next_flush = None
        while True:
            timeout = None
            if pending:
                timeout = max(0.0, (next_flush or 0.0) - time.monotonic())
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                item = None
            if item is _QUEUE_SENTINEL:
                self._flush_batch(pending)
                return
            if item is None:
                if pending:
                    self._flush_batch(pending)
                    pending = []
                    prompt_tokens = 0
                    completion_tokens = 0
                    next_flush = None
                continue
            if pending:
                forecast_prompt = prompt_tokens + item.prompt_tokens
                forecast_completion = completion_tokens + item.completion_tokens
                if (
                    forecast_prompt > self._prompt_limit
                    or forecast_completion > self._completion_limit
                    or len(pending) >= 4
                ):
                    self._flush_batch(pending)
                    pending = []
                    prompt_tokens = 0
                    completion_tokens = 0
                    next_flush = None
            pending.append(item)
            prompt_tokens += item.prompt_tokens
            completion_tokens += item.completion_tokens
            if len(pending) == 1:
                next_flush = time.monotonic() + self._flush_interval
            should_flush = (
                prompt_tokens >= self._prompt_limit
                or completion_tokens >= self._completion_limit
                or len(pending) >= 4
            )
            if should_flush:
                self._flush_batch(pending)
                pending = []
                prompt_tokens = 0
                completion_tokens = 0
                next_flush = None

    def _flush_batch(self, tasks: List[AnalysisChunkTask]) -> None:
        if not tasks:
            return
        try:
            result_map = self._execute_with_backoff(tasks)
        except Exception as exc:  # pragma: no cover - defensive
            for task in tasks:
                task.future.set_exception(exc)
            return
        for task in tasks:
            payload = result_map.get(task.chunk_id)
            if payload is None:
                task.future.set_result(
                    self._processor._normalize_analysis("")
                )
            else:
                task.future.set_result(payload)

    def _execute_with_backoff(self, tasks: List[AnalysisChunkTask]) -> dict:
        try:
            return self._execute(tasks)
        except LmStudioError as exc:
            if self._processor._adjust_token_budget_for_credit_limit(exc, "analysis"):
                return self._execute_with_backoff(tasks)
            if not self._should_split(exc, tasks):
                raise
            first, second = self._split(tasks)
            self._processor._log(
                "warning",
                (
                    f"Packed analysis batch exceeded context window ({exc}). "
                    f"Retrying as {len(first)} + {len(second)} chunk(s)."
                ),
            )
            combined: dict[str, dict] = {}
            combined.update(self._execute_with_backoff(first))
            combined.update(self._execute_with_backoff(second))
            return combined

    def _should_split(self, exc: Exception, tasks: Sequence[AnalysisChunkTask]) -> bool:
        return len(tasks) > 1 and _is_context_limit_error(exc)

    def _split(self, tasks: List[AnalysisChunkTask]) -> tuple[List[AnalysisChunkTask], List[AnalysisChunkTask]]:
        midpoint = max(1, len(tasks) // 2)
        return tasks[:midpoint], tasks[midpoint:]

    def _execute(self, tasks: List[AnalysisChunkTask]) -> dict:
        system_prompt = (
            ANALYSIS_PROMPT
            + " Respond with a JSON object keyed by chunk_id. Never include prose outside JSON."
        )
        blocks = []
        for task in tasks:
            blocks.append(
                "\n".join(
                    [
                        f"chunk_id: {task.chunk_id}",
                        f"language_hint: {task.language_hint}",
                        f"chunk_position: {task.chunk_index} of {task.chunk_total} ({task.document_name})",
                        "content:",
                        task.snippet,
                    ]
                )
            )
        user_prompt = (
            "Process each transcript chunk independently. "
            "For every chunk, return JSON with keys "
            '["summary","topics","tone","keywords","action_items","quality_flags"]. '
            "Return a single JSON object mapping chunk_id to its JSON payload.\n\n"
            + "\n\n".join(blocks)
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw = self._processor._chat_completion(messages)
        payload = LmStudioClient._extract_json_payload(raw) or _safe_json_loads(raw)
        if isinstance(payload, dict):
            normalized = {}
            for chunk_id, value in payload.items():
                if isinstance(value, dict):
                    normalized[chunk_id] = value
                else:
                    normalized[chunk_id] = self._processor._normalize_analysis(str(value))
            return normalized
        return {}


class RewritePackingQueue:
    def __init__(
        self,
        processor: "TranscriptBatchProcessor",
        mode: str,
        prompt_limit: int,
        completion_limit: int,
        flush_interval: float = 0.3,
        token_counter: Optional[Callable[[str], int]] = None,
    ):
        self._processor = processor
        self._mode = mode
        self._prompt_limit = prompt_limit
        self._completion_limit = completion_limit
        self._flush_interval = flush_interval
        self._token_counter = token_counter or estimate_tokens
        self._queue: "queue.Queue[RewriteChunkTask | object]" = queue.Queue()
        self._stop_requested = threading.Event()
        self._worker = threading.Thread(
            target=self._run,
            name=f"rewrite-packer-{mode}",
            daemon=True,
        )
        self._worker.start()

    def submit(self, lines: List[str]) -> Future:
        future: Future = Future()
        prompt_tokens = sum(self._token_counter(line) for line in lines) + 16
        completion_tokens = max(128, prompt_tokens)
        task = RewriteChunkTask(
            lines=lines,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            future=future,
        )
        self._queue.put(task)
        return future

    def close(self) -> None:
        self._stop_requested.set()
        self._queue.put(_QUEUE_SENTINEL)
        self._worker.join(timeout=2.0)

    def update_limits(self, prompt_limit: int, completion_limit: int) -> None:
        self._prompt_limit = max(_MIN_PROMPT_TOKENS, prompt_limit)
        self._completion_limit = max(_MIN_COMPLETION_TOKENS, completion_limit)

    def _run(self) -> None:
        pending: List[RewriteChunkTask] = []
        prompt_tokens = 0
        completion_tokens = 0
        next_flush = None
        while True:
            timeout = None
            if pending:
                timeout = max(0.0, (next_flush or 0.0) - time.monotonic())
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                item = None
            if item is _QUEUE_SENTINEL:
                self._flush_batch(pending)
                return
            if item is None:
                if pending:
                    self._flush_batch(pending)
                    pending = []
                    prompt_tokens = 0
                    completion_tokens = 0
                    next_flush = None
                continue
            if pending:
                forecast_prompt = prompt_tokens + item.prompt_tokens
                forecast_completion = completion_tokens + item.completion_tokens
                if (
                    forecast_prompt > self._prompt_limit
                    or forecast_completion > self._completion_limit
                    or len(pending) >= 6
                ):
                    self._flush_batch(pending)
                    pending = []
                    prompt_tokens = 0
                    completion_tokens = 0
                    next_flush = None
            pending.append(item)
            prompt_tokens += item.prompt_tokens
            completion_tokens += item.completion_tokens
            if len(pending) == 1:
                next_flush = time.monotonic() + self._flush_interval
            should_flush = (
                prompt_tokens >= self._prompt_limit
                or completion_tokens >= self._completion_limit
                or len(pending) >= 6
            )
            if should_flush:
                self._flush_batch(pending)
                pending = []
                prompt_tokens = 0
                completion_tokens = 0
                next_flush = None

    def _flush_batch(self, tasks: List[RewriteChunkTask]) -> None:
        if not tasks:
            return
        try:
            output = self._execute_with_backoff(tasks)
        except Exception as exc:  # pragma: no cover - defensive
            for task in tasks:
                task.future.set_exception(exc)
            return
        offset = 0
        for task in tasks:
            count = len(task.lines)
            rewritten = output[offset : offset + count]
            offset += count
            task.future.set_result(rewritten)

    def _execute_with_backoff(self, tasks: List[RewriteChunkTask]) -> List[str]:
        try:
            return self._execute(tasks)
        except LmStudioError as exc:
            if self._processor._adjust_token_budget_for_credit_limit(exc, f"{self._mode} rewrite"):
                return self._execute_with_backoff(tasks)
            if not self._should_split(exc, tasks):
                raise
            first, second = self._split(tasks)
            self._processor._log(
                "warning",
                (
                    f"Packed rewrite batch exceeded context window ({exc}). "
                    f"Retrying as {len(first)} + {len(second)} chunk(s)."
                ),
            )
            combined: List[str] = []
            combined.extend(self._execute_with_backoff(first))
            combined.extend(self._execute_with_backoff(second))
            return combined

    def _should_split(self, exc: Exception, tasks: Sequence[RewriteChunkTask]) -> bool:
        return len(tasks) > 1 and _is_context_limit_error(exc)

    def _split(self, tasks: List[RewriteChunkTask]) -> tuple[List[RewriteChunkTask], List[RewriteChunkTask]]:
        midpoint = max(1, len(tasks) // 2)
        return tasks[:midpoint], tasks[midpoint:]

    def _execute(self, tasks: List[RewriteChunkTask]) -> List[str]:
        batched: List[str] = []
        for task in tasks:
            batched.extend(task.lines)
        return self._processor._rewrite_batch_call(batched, self._mode)


class TranscriptBatchProcessor:
    def __init__(self, request: BatchRequest, log_callback: Optional[LogCallback] = None):
        self.request = request
        self._log_callback = log_callback
        self._client: Optional[LmStudioClient] = None
        self._prompt_budget: int = 0
        self._completion_budget: int = 0
        self._token_budget_ceiling: int = 0
        self.output_dir = request.output_dir
        self.analysis_dir = self.output_dir / "analysis"
        self.rewrite_dir = self.output_dir / "rewrites"
        self.reports_dir = self.output_dir / "reports"
        self.story_dir = self.output_dir / "stories"
        self.story_meta_dir = self.story_dir / "context"
        self.logs_dir = self.output_dir / "logs"
        self.temp_dir = self.output_dir / ".temp_batches"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = self.logs_dir / f"batch_{timestamp}.log"
        for folder in (
            self.output_dir,
            self.analysis_dir,
            self.rewrite_dir,
            self.reports_dir,
            self.story_dir,
            self.story_meta_dir,
            self.logs_dir,
            self.temp_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)
        self._init_token_budgets()
        configure_token_counter(build_token_counter(self.request.model))
        self._client_lock = threading.Lock()
        self._analysis_queue: Optional[AnalysisPackingQueue] = None
        self._rewrite_queues: dict[str, RewritePackingQueue] = {}
        self._prompt_packing_enabled = bool(getattr(self.request, "prompt_packing", False))
        if self._prompt_packing_enabled:
            self._init_prompt_packing()

    @property
    def log_path(self) -> Path:
        return self._log_path

    def _log(self, level: str, message: str) -> None:
        self._append_log_line(level, message)
        if self._log_callback:
            try:
                self._log_callback(level, message)
            except Exception:  # pragma: no cover
                logger.exception("Failed to emit log callback.")
        else:
            getattr(logger, level, logger.info)(message)

    def _append_log_line(self, level: str, message: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        line = f"{timestamp} [{level.upper()}] {message}\n"
        try:
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        except OSError:
            logger.exception("Failed to write batch log.")

    def _retry_on_credit_limit(self, action: Callable[[], T], *, context: str) -> T:
        while True:
            try:
                return action()
            except LmStudioError as exc:
                if not self._adjust_token_budget_for_credit_limit(exc, context):
                    raise

    def _adjust_token_budget_for_credit_limit(self, exc: Exception, context: str) -> bool:
        limit = _extract_credit_limit(exc)
        if not self._apply_credit_budget_limit(limit):
            return False
        reason = f"limit of {limit} tokens" if limit else "credit restrictions"
        self._log(
            "warning",
            (
                f"{context.capitalize()} requests now use {self._prompt_budget + self._completion_budget} "
                f"tokens per call to satisfy LM Studio {reason}."
            ),
        )
        return True

    def _apply_credit_budget_limit(self, limit_hint: Optional[int]) -> bool:
        current_total = self._prompt_budget + self._completion_budget
        minimum_total = _MIN_PROMPT_TOKENS + _MIN_COMPLETION_TOKENS
        if current_total <= minimum_total:
            return False
        if limit_hint is not None and limit_hint > 0:
            if limit_hint >= current_total:
                return False
            target_total = max(minimum_total, limit_hint)
        else:
            reduction = max(256, current_total // 8)
            target_total = max(minimum_total, current_total - reduction)
        if target_total >= current_total:
            return False
        prompt_capacity = max(_MIN_PROMPT_TOKENS, target_total - _MIN_COMPLETION_TOKENS)
        new_prompt = min(self._prompt_budget, prompt_capacity)
        new_completion = max(_MIN_COMPLETION_TOKENS, target_total - new_prompt)
        if new_prompt + new_completion > target_total:
            new_prompt = max(_MIN_PROMPT_TOKENS, target_total - new_completion)
        new_total = new_prompt + new_completion
        if new_total >= current_total:
            return False
        self._prompt_budget = new_prompt
        self._completion_budget = new_completion
        self._token_budget_ceiling = new_total
        self._update_packing_limits()
        return True

    def _update_packing_limits(self) -> None:
        if self._analysis_queue:
            self._analysis_queue.update_limits(self._analysis_prompt_budget(), self._completion_budget)
        for queue in self._rewrite_queues.values():
            queue.update_limits(self._prompt_budget, self._completion_budget)

    def _init_token_budgets(self) -> None:
        requested_prompt = int(self.request.prompt_token_limit or 0)
        if requested_prompt <= 0:
            requested_prompt = 8192
        requested_prompt = max(_MIN_PROMPT_TOKENS, requested_prompt)
        original_prompt_request = requested_prompt
        context_limit = _model_context_limit(self.request.model)
        context_cap = max(
            _MIN_PROMPT_TOKENS + _MIN_COMPLETION_TOKENS,
            context_limit - _CONTEXT_SAFETY_MARGIN,
        )
        max_prompt_for_context = max(_MIN_PROMPT_TOKENS, context_cap - _MIN_COMPLETION_TOKENS)
        safe_prompt = min(requested_prompt, max_prompt_for_context)
        available_for_completion = max(_MIN_COMPLETION_TOKENS, context_cap - safe_prompt)
        self._prompt_budget = safe_prompt
        self._completion_budget = available_for_completion
        self._token_budget_ceiling = self._prompt_budget + self._completion_budget
        if safe_prompt < original_prompt_request:
            self._log(
                "warning",
                (
                    f"Prompt limit {original_prompt_request} tokens leaves insufficient room for completions. "
                    f"Using {safe_prompt} tokens per request."
                ),
            )

    def _init_prompt_packing(self) -> None:
        operations = self.request.operations
        if ProcessingMode.ANALYSIS in operations:
            self._analysis_queue = AnalysisPackingQueue(
                self,
                prompt_limit=self._analysis_prompt_budget(),
                completion_limit=self._completion_budget,
            )
        if ProcessingMode.REWRITE in operations:
            self._rewrite_queues["polish"] = RewritePackingQueue(
                self,
                mode="polish",
                prompt_limit=self._prompt_budget,
                completion_limit=self._completion_budget,
                token_counter=self._rewrite_line_token_cost,
            )
        if ProcessingMode.DEEP_REWRITE in operations:
            self._rewrite_queues["deep"] = RewritePackingQueue(
                self,
                mode="deep",
                prompt_limit=self._prompt_budget,
                completion_limit=self._completion_budget,
                token_counter=self._rewrite_line_token_cost,
            )

    def _shutdown_prompt_packing(self) -> None:
        if self._analysis_queue:
            self._analysis_queue.close()
            self._analysis_queue = None
        for queue in self._rewrite_queues.values():
            queue.close()
        self._rewrite_queues.clear()

    def _document_slug(self, document: TranscriptDocument) -> str:
        base = re.sub(r"[^0-9A-Za-z_.-]+", "_", document.path.stem)
        if not base:
            base = "document"
        resolved = str(document.path.resolve(strict=False))
        digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:8]
        return f"{base}_{digest}"

    def _document_temp_root(self, document: TranscriptDocument) -> Path:
        return self.temp_dir / self._document_slug(document)

    def _document_snapshot(self, document: TranscriptDocument) -> tuple[int, int]:
        try:
            stat = document.path.stat()
        except OSError:
            return 0, document.stats.character_count
        mtime = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))
        size = getattr(stat, "st_size", 0)
        return mtime, size

    def _operation_temp_paths(self, document: TranscriptDocument, operation: str) -> tuple[Path, Path, Path, Path]:
        root = self._document_temp_root(document) / operation
        chunks_dir = root / "chunks"
        results_dir = root / "results"
        manifest_path = root / "manifest.json"
        return root, chunks_dir, results_dir, manifest_path

    def _reset_operation_storage(self, path: Path) -> None:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    def _read_manifest(self, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return {}
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            return {}
        return {}

    def _write_json_atomic(self, path: Path, payload: Any, *, indent: Optional[int] = 2) -> None:
        tmp = path.with_suffix(path.suffix + f".tmp_{uuid.uuid4().hex}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=indent), encoding="utf-8")
        tmp.replace(path)

    def _write_text_atomic(self, path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + f".tmp_{uuid.uuid4().hex}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)

    def _manifest_is_current(
        self,
        manifest: dict,
        *,
        mtime: int,
        size: int,
        signature: str,
        chunks_dir: Path,
    ) -> bool:
        if not manifest:
            return False
        if manifest.get("source_mtime") != mtime or manifest.get("source_size") != size:
            return False
        if manifest.get("signature") != signature:
            return False
        chunks = manifest.get("chunks")
        if not isinstance(chunks, list):
            return False
        for item in chunks:
            if not isinstance(item, dict):
                return False
            name = item.get("file")
            if not name:
                return False
            if not (chunks_dir / name).exists():
                return False
        return True

    def _prepare_operation_storage(
        self,
        document: TranscriptDocument,
        *,
        operation: str,
        signature: str,
        builder: Callable[[Path], List[dict]],
    ) -> tuple[List[dict], Path, Path]:
        root, chunks_dir, results_dir, manifest_path = self._operation_temp_paths(document, operation)
        mtime, size = self._document_snapshot(document)
        manifest = self._read_manifest(manifest_path)
        if not self._manifest_is_current(manifest, mtime=mtime, size=size, signature=signature, chunks_dir=chunks_dir):
            self._reset_operation_storage(root)
            for folder in (root, chunks_dir, results_dir):
                folder.mkdir(parents=True, exist_ok=True)
            chunks = builder(chunks_dir)
            manifest = {
                "document": str(document.path),
                "operation": operation,
                "source_mtime": mtime,
                "source_size": size,
                "signature": signature,
                "chunks": chunks,
            }
            self._write_json_atomic(manifest_path, manifest)
        else:
            for folder in (root, chunks_dir, results_dir):
                folder.mkdir(parents=True, exist_ok=True)
            chunks = manifest.get("chunks", [])
        return chunks, chunks_dir, results_dir

    def _chunk_payload_path(self, chunks_dir: Path, chunk_id: str, suffix: str) -> Path:
        safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", chunk_id) or "chunk"
        return chunks_dir / f"{safe}{suffix}"

    def _chunk_result_path(self, results_dir: Path, chunk_id: str, suffix: str = ".json") -> Path:
        return self._chunk_payload_path(results_dir, chunk_id, suffix)

    def _cleanup_document_temp(self, document: TranscriptDocument) -> None:
        root = self._document_temp_root(document)
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)

    def _needs_llm(self) -> bool:
        return any(
            mode
            in {ProcessingMode.ANALYSIS, ProcessingMode.REWRITE, ProcessingMode.DEEP_REWRITE, ProcessingMode.STORY}
            for mode in self.request.operations
        )

    def _build_client(self) -> LmStudioClient:
        base_url = (self.request.base_url or "").strip()
        model = (self.request.model or "").strip()
        if not base_url or not model:
            raise ValueError("LM Studio base URL and model must be specified.")
        settings = LmStudioSettings(
            base_url=base_url,
            model=model,
            api_key=self.request.api_key or "",
            max_completion_tokens=None,
            preset_id=self.request.preset_id or "1",
            provider_preferences=self.request.provider_preferences or None,
            reasoning_effort=self.request.reasoning_effort or None,
        )
        return LmStudioClient(settings)

    def _chat_completion(self, messages: List[dict], max_tokens: Optional[int] = None) -> str:
        with self._client_lock:
            return self.client.chat_completion(messages, max_tokens=max_tokens)

    def _rewrite_batch_call(self, lines: List[str], mode: str) -> List[str]:
        language_hint = self.request.language_hint or "auto"
        with self._client_lock:
            return self.client.rewrite_batch(lines, language_hint, mode)

    @property
    def client(self) -> LmStudioClient:
        if self._client is None:
            self._client = self._build_client()
            self._log("info", f"Connecting to LM Studio at {self.request.base_url}")
            self._client.ensure_model_loaded(
                lambda level, msg: self._log(level, msg),
                timeout=600.0,
            )
        return self._client

    def iter_files(self) -> List[Path]:
        files = discover_transcripts(
            self.request.input_dir,
            recursive=self.request.recursive,
            ignore_output=self.request.output_dir,
        )
        if self.request.max_files:
            return files[: self.request.max_files]
        return files

    def process_iter(self, files: Optional[Sequence[Path]] = None) -> Iterator[tuple[Path, Optional[FileResult], Optional[Exception], int, int]]:
        target_files = list(files) if files is not None else self.iter_files()
        total = len(target_files)
        if not total:
            self._log("warning", "No transcripts found for processing.")
            self._shutdown_prompt_packing()
            return
        self._log("info", f"Processing {total} transcript(s).")
        if self._needs_llm():
            _ = self.client  # ensure initialization
        try:
            if self._prompt_packing_enabled:
                yield from self._process_iter_parallel(target_files, total)
            else:
                yield from self._process_iter_sequential(target_files, total)
        finally:
            self._shutdown_prompt_packing()

    def _process_iter_sequential(self, target_files: Sequence[Path], total: int) -> Iterator[tuple[Path, Optional[FileResult], Optional[Exception], int, int]]:
        for idx, path in enumerate(target_files, start=1):
            if getattr(self.request, "resume", True) and self._should_skip_file(path):
                self._log("info", f"[{idx}/{total}] {path.name} already processed; skipping.")
                continue
            self._log("info", f"[{idx}/{total}] {path.name}")
            try:
                document = load_transcript(path)
                result = self._process_document(document)
                yield path, result, None, idx, total
            except (LmStudioError, ValueError, OSError) as exc:
                self._log("error", f"{path.name}: {exc}")
                yield path, None, exc, idx, total
            except Exception as exc:  # pragma: no cover - defensive
                self._log("error", f"Unexpected error for {path}: {exc}")
                yield path, None, exc, idx, total

    def _process_iter_parallel(self, target_files: Sequence[Path], total: int) -> Iterator[tuple[Path, Optional[FileResult], Optional[Exception], int, int]]:
        workers = min(max(2, (os.cpu_count() or 4) // 2), total)
        self._log("info", f"Prompt packing enabled; using {workers} worker(s).")
        skipped: set[int] = set()
        futures: dict[Future, tuple[int, Path]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for idx, path in enumerate(target_files, start=1):
                if getattr(self.request, "resume", True) and self._should_skip_file(path):
                    self._log("info", f"[{idx}/{total}] {path.name} already processed; skipping.")
                    skipped.add(idx)
                    continue
                self._log("info", f"[{idx}/{total}] {path.name}")
                future = executor.submit(self._process_single_path, path)
                futures[future] = (idx, path)
            if not futures:
                return
            pending: dict[int, tuple[Path, Optional[FileResult], Optional[Exception], int, int]] = {}
            next_ready = 1
            for future in as_completed(futures):
                idx, path = futures[future]
                try:
                    result = future.result()
                    pending[idx] = (path, result, None, idx, total)
                except (LmStudioError, ValueError, OSError) as exc:
                    self._log("error", f"{path.name}: {exc}")
                    pending[idx] = (path, None, exc, idx, total)
                except Exception as exc:  # pragma: no cover - defensive
                    self._log("error", f"Unexpected error for {path}: {exc}")
                    pending[idx] = (path, None, exc, idx, total)
                while True:
                    if next_ready in skipped:
                        next_ready += 1
                        continue
                    payload = pending.pop(next_ready, None)
                    if payload is None:
                        break
                    yield payload
                    next_ready += 1
            # flush any remaining (handles cases where skips are at the end)
            while True:
                if next_ready in skipped:
                    next_ready += 1
                    continue
                payload = pending.pop(next_ready, None)
                if payload is None:
                    break
                yield payload
                next_ready += 1

    def _process_single_path(self, path: Path) -> FileResult:
        document = load_transcript(path)
        return self._process_document(document)

    def run(self, files: Optional[Sequence[Path]] = None) -> BatchReport:
        report = BatchReport(self.request)
        for path, result, error, _, _ in self.process_iter(files):
            if error is None and result is not None:
                report.record_success(result)
            else:
                reason = str(error) if error else "Unknown error"
                report.record_failure(path, reason)
        report.complete()
        self._write_report(report)
        self._merge_story_outputs(report)
        return report

    def _should_skip_file(self, path: Path) -> bool:
        expected = self._expected_artifacts(path)
        if not expected:
            return False
        for target in expected:
            if not target.exists():
                return False
        return True

    def _expected_artifacts(self, source: Path) -> List[Path]:
        outputs: List[Path] = []
        stem = source.stem
        operations = self.request.operations
        if ProcessingMode.ANALYSIS in operations:
            outputs.append(self.analysis_dir / f"{stem}_analysis.json")
        if ProcessingMode.REWRITE in operations:
            outputs.append(self._rewrite_output_path(source, suffix="polish"))
        if ProcessingMode.DEEP_REWRITE in operations:
            outputs.append(self._rewrite_output_path(source, suffix="deep"))
        if ProcessingMode.STORY in operations:
            outputs.append(self.story_dir / f"{stem}_story.md")
            outputs.append(self.story_meta_dir / f"{stem}_story_meta.json")
        return outputs

    def _rewrite_output_path(self, source: Path, *, suffix: str) -> Path:
        extension = ".srt" if source.suffix.lower() == ".srt" else ".txt"
        return self.rewrite_dir / f"{source.stem}_{suffix}{extension}"

    def _process_document(self, document: TranscriptDocument) -> FileResult:
        result = FileResult(document=document, stats=document.stats)
        if ProcessingMode.ANALYSIS in self.request.operations:
            self._log("info", f"Starting analysis for {document.short_name()}.")
            analysis_path, excerpt = self._run_analysis(document)
            result.analysis_path = analysis_path
            result.summary_excerpt = excerpt
        if ProcessingMode.REWRITE in self.request.operations:
            self._log("info", f"Running polish rewrite for {document.short_name()}.")
            rewrite_path = self._rewrite(document, mode="polish")
            result.rewrite_paths[ProcessingMode.REWRITE.value] = rewrite_path
        if ProcessingMode.DEEP_REWRITE in self.request.operations:
            self._log("info", f"Running deep rewrite for {document.short_name()}.")
            deep_path = self._rewrite(document, mode="deep")
            result.rewrite_paths[ProcessingMode.DEEP_REWRITE.value] = deep_path
        if ProcessingMode.STORY in self.request.operations:
            self._log("info", f"Building narrative story for {document.short_name()}.")
            story_path, meta_path = self._build_story(document, result)
            result.story_path = story_path
            result.story_meta_path = meta_path
        self._cleanup_document_temp(document)
        return result

    def _run_analysis(self, document: TranscriptDocument) -> tuple[Path, Optional[str]]:
        def execute() -> tuple[Path, Optional[str]]:
            chunk_meta, chunks_dir, results_dir = self._prepare_analysis_chunks(document)
            total_chunks = len(chunk_meta)
            if total_chunks > 1:
                self._log("info", f"Transcript too long for a single prompt; analyzing in {total_chunks} chunks.")
            lang_hint = self.request.language_hint or "auto"
            total_chunks = max(1, total_chunks)

            def _pending_chunks() -> List[dict]:
                missing: List[dict] = []
                for meta in chunk_meta:
                    chunk_id = meta.get("chunk_id")
                    if not chunk_id:
                        continue
                    cached = self._load_analysis_chunk_result(results_dir, chunk_id)
                    if cached is None:
                        missing.append(meta)
                return missing

            pending = _pending_chunks()
            cached_count = total_chunks - len(pending)
            if cached_count and pending:
                self._log(
                    "info",
                    (
                        f"Resuming analysis for {document.short_name()}: "
                        f"{cached_count} chunk(s) already completed."
                    ),
                )
            if pending:
                if self._analysis_queue:
                    futures: List[tuple[dict, Future]] = []
                    for meta in pending:
                        snippet = self._load_analysis_chunk(chunks_dir, meta)
                        futures.append(
                            (
                                meta,
                                self._analysis_queue.submit(
                                    snippet,
                                    language_hint=lang_hint,
                                    chunk_index=int(meta.get("index") or 1),
                                    chunk_total=int(meta.get("chunk_total") or total_chunks),
                                    document_name=document.short_name(),
                                    chunk_id=str(meta.get("chunk_id")),
                                ),
                            )
                        )
                    for meta, future in futures:
                        try:
                            result = future.result()
                        except Exception as exc:
                            raise LmStudioError(f"Packed analysis failed: {exc}") from exc
                        if not isinstance(result, dict):
                            result = self._normalize_analysis("")
                        self._store_analysis_chunk_result(results_dir, str(meta.get("chunk_id")), result)
                else:
                    for meta in pending:
                        snippet = self._load_analysis_chunk(chunks_dir, meta)
                        prompt = self._build_analysis_prompt(
                            snippet,
                            lang_hint,
                            int(meta.get("index") or 1),
                            int(meta.get("chunk_total") or total_chunks),
                        )
                        raw = self._chat_completion([{"role": "user", "content": prompt}])
                        payload = self._normalize_analysis(raw)
                        self._store_analysis_chunk_result(results_dir, str(meta.get("chunk_id")), payload)

            chunk_results: List[dict[str, Any]] = []
            for meta in chunk_meta:
                payload = self._load_analysis_chunk_result(results_dir, str(meta.get("chunk_id")))
                if payload is None:
                    payload = self._normalize_analysis("")
                chunk_results.append(payload)
            if not chunk_results:
                chunk_results.append(self._normalize_analysis(""))
            if len(chunk_results) == 1:
                analysis = chunk_results[0]
            else:
                analysis = self._merge_analysis_chunks(chunk_results)
                self._log("info", f"Merged {len(chunk_results)} chunk analyses into a single report.")
            output_path = self.analysis_dir / f"{document.path.stem}_analysis.json"
            output_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
            excerpt = None
            summary = analysis.get("summary")
            if isinstance(summary, str):
                excerpt = summary[:240]
            return output_path, excerpt

        return self._retry_on_credit_limit(execute, context="analysis")

    def _normalize_analysis(self, raw: str) -> dict:
        candidate = raw.strip()
        if not candidate:
            return {"summary": "", "topics": [], "tone": "", "keywords": [], "action_items": [], "quality_flags": []}
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        return {
            "summary": candidate,
            "topics": [],
            "tone": "",
            "keywords": [],
            "action_items": [],
            "quality_flags": [],
        }

    def _analysis_chunks(self, document: TranscriptDocument) -> List[Tuple[int, int]]:
        if not document.lines:
            return [(0, 0)]
        prompt_limit = self._analysis_prompt_budget()
        margin = self._analysis_margin(prompt_limit)
        token_limit = max(1, prompt_limit - margin)
        ranges: List[Tuple[int, int]] = []
        start_index = 0
        bucket_tokens = 0
        bucket_count = 0
        for idx, line in enumerate(document.lines):
            token_cost = estimate_tokens(line)
            exceeds_tokens = (bucket_tokens + token_cost) > token_limit
            if bucket_count > 0 and exceeds_tokens:
                ranges.append((start_index, idx))
                start_index = idx
                bucket_tokens = 0
                bucket_count = 0
            bucket_count += 1
            bucket_tokens += token_cost
        if bucket_count > 0:
            ranges.append((start_index, len(document.lines)))
        else:
            ranges.append((0, 0))
        return ranges

    def _analysis_prompt_budget(self) -> int:
        return max(_MIN_PROMPT_TOKENS, self._prompt_budget)

    def _analysis_margin(self, prompt_limit: int) -> int:
        desired = DEFAULT_LMSTUDIO_TOKEN_MARGIN
        max_margin = max(128, prompt_limit - 256)
        return min(max(128, desired), max_margin)

    def _build_analysis_prompt(self, snippet: str, lang_hint: str, chunk_index: int, chunk_total: int) -> str:
        chunk_note = (
            f"Language hint: {lang_hint}. "
            f"Chunk {chunk_index} of {chunk_total}. "
        )
        if chunk_total > 1:
            chunk_note += "Focus on this portion; other chunks will be summarized separately before merging."
        else:
            chunk_note += "This chunk contains the full transcript."
        return f"{ANALYSIS_PROMPT}\n\n{chunk_note}\n\nTranscript chunk:\n{snippet}"

    def _merge_analysis_chunks(self, analyses: Sequence[dict]) -> dict:
        summaries = [str(item.get("summary") or "").strip() for item in analyses]
        merged = {
            "summary": self._merge_summaries(summaries),
            "topics": self._merge_string_lists(analyses, "topics", 18),
            "tone": self._merge_tones(analyses),
            "keywords": self._merge_string_lists(analyses, "keywords", 24),
            "action_items": self._merge_string_lists(analyses, "action_items", 20),
            "quality_flags": self._merge_string_lists(analyses, "quality_flags", 20),
        }
        return merged

    def _prepare_analysis_chunks(self, document: TranscriptDocument) -> tuple[List[dict], Path, Path]:
        prompt_limit = self._analysis_prompt_budget()
        margin = self._analysis_margin(prompt_limit)
        signature = f"analysis|prompt={prompt_limit}|margin={margin}|lines={len(document.lines)}"

        def builder(chunks_dir: Path) -> List[dict]:
            ranges = self._analysis_chunks(document)
            total = len(ranges)
            slug = self._document_slug(document)
            chunk_meta: List[dict] = []
            for idx, (start, end) in enumerate(ranges, start=1):
                chunk_id = f"{slug}_analysis_{idx:04d}"
                payload_path = self._chunk_payload_path(chunks_dir, chunk_id, ".txt")
                snippet = "\n".join(document.lines[start:end])
                self._write_text_atomic(payload_path, snippet)
                chunk_meta.append(
                    {
                        "index": idx,
                        "chunk_id": chunk_id,
                        "file": payload_path.name,
                        "start": start,
                        "end": end,
                        "chunk_total": total or 1,
                    }
                )
            if not chunk_meta:
                chunk_id = f"{slug}_analysis_0001"
                payload_path = self._chunk_payload_path(chunks_dir, chunk_id, ".txt")
                self._write_text_atomic(payload_path, "")
                chunk_meta.append(
                    {
                        "index": 1,
                        "chunk_id": chunk_id,
                        "file": payload_path.name,
                        "start": 0,
                        "end": 0,
                        "chunk_total": 1,
                    }
                )
            return chunk_meta

        return self._prepare_operation_storage(
            document,
            operation="analysis",
            signature=signature,
            builder=builder,
        )

    def _load_analysis_chunk(self, chunks_dir: Path, meta: dict) -> str:
        name = meta.get("file")
        if not name:
            return ""
        path = chunks_dir / name
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"Missing analysis chunk payload: {path}") from exc

    def _store_analysis_chunk_result(self, results_dir: Path, chunk_id: str, payload: dict) -> None:
        path = self._chunk_result_path(results_dir, chunk_id)
        self._write_json_atomic(path, payload)

    def _load_analysis_chunk_result(self, results_dir: Path, chunk_id: str) -> Optional[dict]:
        path = self._chunk_result_path(results_dir, chunk_id)
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            self._log("warning", f"Failed to read cached analysis chunk {path.name}; it will be regenerated.")
        return None

    def _merge_summaries(self, summaries: Sequence[str], max_sentences: int = 10) -> str:
        sentences: List[str] = []
        for summary in summaries:
            if not summary:
                continue
            for sentence in _SENTENCE_SPLIT_RE.split(summary):
                clean = sentence.strip()
                if clean:
                    sentences.append(clean)
        if not sentences:
            return ""
        clipped = sentences[:max_sentences]
        text = " ".join(clipped)
        return text.strip()

    def _merge_string_lists(self, analyses: Sequence[dict], field: str, max_items: int) -> List[str]:
        merged: List[str] = []
        seen: set[str] = set()
        for analysis in analyses:
            values = analysis.get(field)
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, str):
                    normalized = value.strip()
                else:
                    normalized = str(value).strip()
                if not normalized:
                    continue
                key = normalized.casefold()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(normalized)
                if len(merged) >= max_items:
                    return merged
        return merged

    def _merge_tones(self, analyses: Sequence[dict]) -> str:
        tones: List[str] = []
        for analysis in analyses:
            tone = analysis.get("tone")
            if isinstance(tone, str):
                value = tone.strip()
                if value:
                    tones.append(value)
        if not tones:
            return ""
        counts: Counter[str] = Counter()
        first_index: dict[str, int] = {}
        for idx, tone in enumerate(tones):
            key = tone.casefold()
            counts[key] += 1
            first_index.setdefault(key, idx)
        best_key = max(counts.keys(), key=lambda key: (counts[key], -first_index[key]))
        return tones[first_index[best_key]]

    def _prepare_rewrite_chunks(self, document: TranscriptDocument, mode: str) -> tuple[List[dict], Path, Path]:
        margin = self._rewrite_margin(self._prompt_budget)
        chunk_size = self._rewrite_chunk_size() or 0
        signature = (
            f"rewrite|mode={mode}|prompt={self._prompt_budget}|margin={margin}|size={chunk_size}|lines={len(document.lines)}"
        )

        def builder(chunks_dir: Path) -> List[dict]:
            slug = self._document_slug(document)
            chunk_meta: List[dict] = []
            line_cursor = 0
            for idx, chunk in enumerate(self._iter_rewrite_chunks(document), start=1):
                chunk_lines = list(chunk)
                chunk_id = f"{slug}_{mode}_{idx:04d}"
                payload_path = self._chunk_payload_path(chunks_dir, chunk_id, ".json")
                self._write_json_atomic(payload_path, chunk_lines, indent=None)
                count = len(chunk_lines)
                chunk_meta.append(
                    {
                        "index": idx,
                        "chunk_id": chunk_id,
                        "file": payload_path.name,
                        "start": line_cursor,
                        "end": line_cursor + count,
                        "size": count,
                    }
                )
                line_cursor += count
            if not chunk_meta:
                chunk_id = f"{slug}_{mode}_0001"
                payload_path = self._chunk_payload_path(chunks_dir, chunk_id, ".json")
                self._write_json_atomic(payload_path, [], indent=None)
                chunk_meta.append(
                    {
                        "index": 1,
                        "chunk_id": chunk_id,
                        "file": payload_path.name,
                        "start": 0,
                        "end": 0,
                        "size": 0,
                    }
                )
            return chunk_meta

        return self._prepare_operation_storage(
            document,
            operation=f"rewrite_{mode}",
            signature=signature,
            builder=builder,
        )

    def _load_rewrite_chunk(self, chunks_dir: Path, meta: dict) -> List[str]:
        name = meta.get("file")
        if not name:
            return []
        path = chunks_dir / name
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(item) for item in data]
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Missing rewrite chunk payload: {path}") from exc
        raise RuntimeError(f"Rewrite chunk payload invalid: {path}")

    def _store_rewrite_chunk_result(self, results_dir: Path, chunk_id: str, lines: Sequence[str]) -> None:
        path = self._chunk_result_path(results_dir, chunk_id)
        self._write_json_atomic(path, list(lines), indent=None)

    def _load_rewrite_chunk_result(self, results_dir: Path, chunk_id: str) -> Optional[List[str]]:
        path = self._chunk_result_path(results_dir, chunk_id)
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(item) for item in data]
        except (OSError, json.JSONDecodeError):
            self._log("warning", f"Failed to read cached rewrite chunk {path.name}; it will be regenerated.")
        return None

    def _collect_rewrite_lines(
        self,
        chunk_meta: Sequence[dict],
        *,
        chunks_dir: Path,
        results_dir: Path,
    ) -> List[str]:
        aggregated: List[str] = []
        for meta in chunk_meta:
            chunk_id = str(meta.get("chunk_id"))
            payload = self._load_rewrite_chunk_result(results_dir, chunk_id)
            if payload is None:
                self._log("warning", f"Rewrite chunk {chunk_id} missing; using original text.")
                payload = self._load_rewrite_chunk(chunks_dir, meta)
            aggregated.extend(payload)
        return aggregated

    def _gather_rewrite_lines(self, document: TranscriptDocument, mode: str) -> List[str]:
        chunk_meta, chunks_dir, results_dir = self._prepare_rewrite_chunks(document, mode)
        total_chunks = len(chunk_meta) or 1

        def pending_chunks() -> List[dict]:
            missing: List[dict] = []
            for meta in chunk_meta:
                chunk_id = meta.get("chunk_id")
                if not chunk_id:
                    continue
                cached = self._load_rewrite_chunk_result(results_dir, str(chunk_id))
                if cached is None:
                    missing.append(meta)
            return missing

        pending = pending_chunks()
        cached_count = total_chunks - len(pending)
        if cached_count and pending:
            self._log(
                "info",
                (
                    f"Resuming {mode} rewrite for {document.short_name()}: "
                    f"{cached_count} chunk(s) already completed."
                ),
            )
        if pending:
            queue = self._rewrite_queues.get(mode)
            if queue:
                futures: List[tuple[dict, Future]] = []
                for meta in pending:
                    payload = self._load_rewrite_chunk(chunks_dir, meta)
                    futures.append((meta, queue.submit(payload)))
                for meta, future in futures:
                    try:
                        rewritten = future.result()
                    except Exception as exc:
                        raise LmStudioError(f"Packed rewrite failed: {exc}") from exc
                    self._store_rewrite_chunk_result(results_dir, str(meta.get("chunk_id")), rewritten)
            else:
                for meta in pending:
                    payload = self._load_rewrite_chunk(chunks_dir, meta)
                    rewritten = self._rewrite_batch_call(payload, mode)
                    self._store_rewrite_chunk_result(results_dir, str(meta.get("chunk_id")), rewritten)
        return self._collect_rewrite_lines(chunk_meta, chunks_dir=chunks_dir, results_dir=results_dir)

    def _rewrite(self, document: TranscriptDocument, mode: str) -> Path:
        def execute() -> Path:
            if document.kind == "srt" and document.srt_entries:
                return self._rewrite_srt(document, mode)
            return self._rewrite_plain(document, mode)

        return self._retry_on_credit_limit(execute, context=f"{mode} rewrite")

    def _rewrite_plain(self, document: TranscriptDocument, mode: str) -> Path:
        suffix = "deep" if mode == "deep" else "polish"
        target = self.rewrite_dir / f"{document.path.stem}_{suffix}.txt"
        rewritten_lines = self._gather_rewrite_lines(document, mode)
        with target.open("w", encoding="utf-8") as handle:
            for idx, line in enumerate(rewritten_lines):
                if idx:
                    handle.write("\n")
                handle.write(line)
        return target

    def _rewrite_srt(self, document: TranscriptDocument, mode: str) -> Path:
        assert document.srt_entries is not None
        suffix = "deep" if mode == "deep" else "polish"
        target = self.rewrite_dir / f"{document.path.stem}_{suffix}.srt"
        entries = document.srt_entries
        total_entries = len(entries)
        entry_index = 0
        exceeded = False
        rewritten_lines = self._gather_rewrite_lines(document, mode)
        with target.open("w", encoding="utf-8") as handle:
            for line in rewritten_lines:
                if entry_index >= total_entries:
                    exceeded = True
                    break
                original = entries[entry_index]
                updated = srt.Subtitle(
                    index=entry_index + 1,
                    start=original.start,
                    end=original.end,
                    content=line,
                    proprietary=original.proprietary,
                )
                handle.write(updated.to_srt())
                entry_index += 1
        if exceeded:
            self._log(
                "warning",
                f"Received more rewritten lines than cues for {document.short_name()}; extra output was discarded.",
            )
        if entry_index < total_entries:
            remaining = total_entries - entry_index
            self._log(
                "warning",
                f"Rewrite output shorter than expected for {document.short_name()}; keeping {remaining} original cue(s).",
            )
            with target.open("a", encoding="utf-8") as handle:
                for idx in range(entry_index, total_entries):
                    original = entries[idx]
                    fallback = srt.Subtitle(
                        index=idx + 1,
                        start=original.start,
                        end=original.end,
                        content=original.content,
                        proprietary=original.proprietary,
                    )
                    handle.write(fallback.to_srt())
        return target

    def _rewrite_chunk_size(self) -> Optional[int]:
        batch_size = getattr(self.request, "batch_size", 0) or 0
        return int(batch_size) if batch_size > 0 else None

    def _rewrite_line_token_cost(self, text: str) -> int:
        serialized = json.dumps(text, ensure_ascii=False)
        return estimate_tokens(serialized) + _REWRITE_JSON_ITEM_OVERHEAD

    def _rewrite_margin(self, prompt_limit: int) -> int:
        base = DEFAULT_LMSTUDIO_TOKEN_MARGIN
        desired = base + _REWRITE_JSON_FIXED_MARGIN
        max_margin = max(256, prompt_limit - 256)
        return min(max(256, desired), max_margin)

    def _iter_rewrite_chunks(self, document: TranscriptDocument) -> Iterator[List[str]]:
        margin = self._rewrite_margin(self._prompt_budget)
        return chunked(
            document.lines,
            size=self._rewrite_chunk_size(),
            max_tokens=self._prompt_budget,
            token_margin=margin,
            token_counter=self._rewrite_line_token_cost,
        )

    def _ensure_analysis_for_story(self, document: TranscriptDocument, result: FileResult) -> tuple[Path, dict]:
        path = result.analysis_path
        if path:
            path = Path(path)
        if path and path.exists():
            analysis = self._read_analysis_payload(path)
        else:
            candidate = self.analysis_dir / f"{document.path.stem}_analysis.json"
            if candidate.exists():
                path = candidate
                analysis = self._read_analysis_payload(candidate)
            else:
                self._log(
                    "info",
                    f"Story mode requires analysis for {document.short_name()}; generating report.",
                )
                path, excerpt = self._run_analysis(document)
                if excerpt:
                    result.summary_excerpt = excerpt
                analysis = self._read_analysis_payload(path)
        result.analysis_path = path
        summary = analysis.get("summary")
        if isinstance(summary, str) and summary.strip() and not result.summary_excerpt:
            result.summary_excerpt = summary.strip()[:240]
        return path, analysis

    def _read_analysis_payload(self, path: Path) -> dict:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            self._log("warning", f"Failed to read analysis file {path}: {exc}")
            return self._normalize_analysis("")
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError as exc:
            self._log("warning", f"Invalid analysis JSON at {path}: {exc}")
        return self._normalize_analysis("")

    def _ensure_story_rewrite_lines(self, document: TranscriptDocument, result: FileResult) -> tuple[Path, List[str]]:
        preference = [
            ("deep", ProcessingMode.DEEP_REWRITE.value),
            ("polish", ProcessingMode.REWRITE.value),
        ]
        for suffix, key in preference:
            path = self._locate_rewrite_path(document, result, suffix=suffix, key=key)
            if path:
                lines = self._story_lines_from_path(path)
                if lines:
                    return path, lines
        preferred_mode = "deep" if ProcessingMode.DEEP_REWRITE in self.request.operations else "polish"
        self._log(
            "info",
            f"Story mode requires a {preferred_mode} rewrite for {document.short_name()}; generating it now.",
        )
        rewrite_path = self._rewrite(document, mode=preferred_mode)
        key = ProcessingMode.DEEP_REWRITE.value if preferred_mode == "deep" else ProcessingMode.REWRITE.value
        result.rewrite_paths[key] = rewrite_path
        lines = self._story_lines_from_path(rewrite_path)
        if not lines:
            lines = [line.strip() for line in document.lines if line.strip()]
            self._log(
                "warning",
                f"Rewrite output empty for {document.short_name()}; falling back to raw transcript lines for story mode.",
            )
        return rewrite_path, lines

    def _locate_rewrite_path(self, document: TranscriptDocument, result: FileResult, *, suffix: str, key: str) -> Optional[Path]:
        existing = result.rewrite_paths.get(key)
        candidate: Optional[Path] = None
        if existing:
            candidate = Path(existing)
        else:
            inferred = self._rewrite_output_path(document.path, suffix=suffix)
            if inferred.exists():
                candidate = inferred
        if candidate and candidate.exists():
            result.rewrite_paths[key] = candidate
            return candidate
        return None

    def _story_lines_from_path(self, path: Path) -> List[str]:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            self._log("warning", f"Failed to read rewrite file {path}: {exc}")
            return []
        if path.suffix.lower() == ".srt":
            try:
                entries = list(srt.parse(text))
                lines: List[str] = []
                for entry in entries:
                    snippet = entry.content.strip()
                    if not snippet:
                        continue
                    for line in snippet.splitlines():
                        clean = line.strip()
                        if clean:
                            lines.append(clean)
                return lines
            except Exception as exc:  # pragma: no cover - safety net for malformed subtitles
                self._log("warning", f"Failed to parse rewrite SRT {path}: {exc}")
        return [line.strip() for line in text.splitlines() if line.strip()]

    def _build_story(self, document: TranscriptDocument, result: FileResult) -> tuple[Path, Path]:
        return self._retry_on_credit_limit(
            lambda: self._build_story_impl(document, result),
            context="story generation",
        )

    def _build_story_impl(self, document: TranscriptDocument, result: FileResult) -> tuple[Path, Path]:
        self._log("info", f"Crafting narrative for {document.short_name()}...")
        lang_hint = self.request.language_hint or "auto"
        language_label = self._language_label(lang_hint)
        story_path = self.story_dir / f"{document.path.stem}_story.md"
        _, analysis_payload = self._ensure_analysis_for_story(document, result)
        rewrite_path, rewrite_lines = self._ensure_story_rewrite_lines(document, result)
        self._log(
            "info",
            f"Story mode will adapt rewrite {rewrite_path.name} using analysis insights.",
        )
        if not rewrite_lines:
            rewrite_lines = [line.strip() for line in document.lines if line.strip()]
        if not rewrite_lines:
            rewrite_lines = ["(Исходный текст пуст.)"]

        summary_value = analysis_payload.get("summary")
        if isinstance(summary_value, str):
            analysis_summary = summary_value.strip()
        else:
            analysis_summary = ""
        if not analysis_summary:
            analysis_summary = "Summary unavailable; rely on the rewrite facts and preserve strict chronology."
        analysis_summary = analysis_summary[:1200]

        def clean_values(values: Any, limit: int) -> List[str]:
            cleaned: List[str] = []
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, str):
                        candidate = value.strip()
                    else:
                        candidate = str(value).strip()
                    if not candidate:
                        continue
                    cleaned.append(candidate)
                    if len(cleaned) >= limit:
                        break
            return cleaned

        def format_block(values: List[str], empty_message: str) -> str:
            if not values:
                return f"- {empty_message}"
            return "\n".join(f"- {value}" for value in values)

        topics_block = format_block(clean_values(analysis_payload.get("topics"), 12), "темы не указаны")
        issues_block = format_block(clean_values(analysis_payload.get("quality_flags"), 12), "замечаний нет")
        actions_block = format_block(clean_values(analysis_payload.get("action_items"), 10), "приоритеты не заданы")

        chunk_iterator = chunked(
            rewrite_lines,
            size=self._rewrite_chunk_size(),
            max_tokens=self._prompt_budget,
            token_margin=self._rewrite_margin(self._prompt_budget),
        )

        tail_window = ""
        meta_excerpt = ""
        wrote_any = False
        with story_path.open("w", encoding="utf-8") as story_file:
            for idx, chunk_lines in enumerate(chunk_iterator, start=1):
                snippet = "\n".join(chunk_lines).strip()
                if not snippet:
                    continue
                previous_excerpt = tail_window.strip()
                user_prompt = STORY_PASS_PROMPT.format(
                    analysis_summary=analysis_summary,
                    topic_bullets=topics_block,
                    issue_bullets=issues_block,
                    action_bullets=actions_block,
                    previous=previous_excerpt or "(Начало исторического изложения.)",
                    chunk=snippet,
                    language=language_label,
                )
                messages = [
                    {"role": "system", "content": STORY_SYSTEM_PROMPT.format(language=language_label)},
                    {"role": "user", "content": user_prompt},
                ]
                raw_story = self.client.chat_completion(messages)
                cleaned = raw_story.strip()
                if not cleaned:
                    self._log("warning", f"Story chunk {idx} returned empty response.")
                    continue
                if wrote_any:
                    story_file.write("\n\n")
                story_file.write(cleaned)
                wrote_any = True
                tail_window = self._append_text_window(tail_window, cleaned, limit=1800)
                meta_excerpt = self._append_text_window(meta_excerpt, cleaned, limit=12000)

        if not wrote_any:
            fallback_text = (
                "История не могла быть сформирована: модель вернула пустой ответ. "
                "Проверьте переписанный текст и повторите попытку."
            )
            story_path.write_text(fallback_text, encoding="utf-8")
            meta_excerpt = fallback_text[-12000:]
        meta = self._build_story_metadata(meta_excerpt, language_label)
        meta_path = self.story_meta_dir / f"{document.path.stem}_story_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return story_path, meta_path

    def _build_story_metadata(self, story_excerpt: str, language_label: str) -> dict:
        if not story_excerpt.strip():
            return EMPTY_STORY_META.copy()
        messages = [
            {"role": "system", "content": STORY_CONTEXT_SYSTEM_PROMPT},
            {"role": "user", "content": STORY_CONTEXT_PROMPT.format(story=story_excerpt, language=language_label)},
        ]
        raw = self.client.chat_completion(messages)
        return self._normalize_story_meta(raw)

    def _normalize_story_meta(self, raw: str) -> dict:
        candidate = raw.strip()
        if not candidate:
            return EMPTY_STORY_META.copy()
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                merged = EMPTY_STORY_META.copy()
                merged.update(data)
                return merged
        except json.JSONDecodeError:
            pass
        fallback = EMPTY_STORY_META.copy()
        fallback["synopsis"] = candidate
        return fallback

    @staticmethod
    def _append_text_window(buffer: str, addition: str, limit: int) -> str:
        if not addition:
            return buffer[-limit:]
        if buffer:
            buffer = f"{buffer}\n\n{addition}"
        else:
            buffer = addition
        if len(buffer) > limit:
            buffer = buffer[-limit:]
        return buffer


    def _write_report(self, report: BatchReport) -> tuple[Path, Path]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.reports_dir / f"batch_report_{timestamp}.json"
        json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        md_path = self.reports_dir / f"batch_report_{timestamp}.md"
        md_path.write_text(self._render_markdown(report), encoding="utf-8")
        return json_path, md_path

    def _merge_story_outputs(self, report: BatchReport) -> Optional[Path]:
        if ProcessingMode.STORY not in self.request.operations:
            return None
        if not self.request.merge_story_output:
            return None
        merged_sections: List[str] = []
        for item in report.processed:
            story_path = item.story_path
            if not story_path:
                continue
            try:
                content = story_path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                self._log("warning", f"Failed to read {story_path}: {exc}")
                continue
            if not content:
                continue
            title = item.document.short_name()
            merged_sections.append(f"## {title}\n\n{content}")
        if not merged_sections:
            self._log("warning", "Story merge enabled but no story outputs were found to combine.")
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        merged_path = self.story_dir / f"merged_story_{timestamp}.md"
        merged_path.write_text("\n\n---\n\n".join(merged_sections), encoding="utf-8")
        self._log("success", f"Merged story saved to {merged_path}")
        return merged_path

    def merge_story_outputs(self, report: BatchReport) -> Optional[Path]:
        """Public helper for callers that manage the run loop manually (GUI)."""
        return self._merge_story_outputs(report)

    def save_report(self, report: BatchReport) -> tuple[Path, Path]:
        """Public helper so external callers (GUI) can persist reports."""
        return self._write_report(report)

    def _render_markdown(self, report: BatchReport) -> str:
        data = report.to_dict()
        summary = data["summary"]
        lines = [
            f"# Transcript Batch Report ({datetime.now().isoformat(timespec='seconds')})",
            "",
            f"- Processed: {summary['processed']}",
            f"- Failed: {summary['failed']}",
            f"- Total files: {summary['total_files']}",
            f"- Total words: {summary['total_words']}",
            f"- Total minutes: {summary['total_minutes']}",
            "",
            "## Files",
        ]
        for item in data["processed"]:
            lines.append(
                f"- **{Path(item['file']).name}** - lines: {item['stats']['line_count']} - words: {item['stats']['word_count']}"
            )
        if data["failed"]:
            lines.append("")
            lines.append("## Failures")
            for path, reason in data["failed"].items():
                lines.append(f"- **{Path(path).name}** - {reason}")
        return "\n".join(lines)

    @staticmethod
    def _language_label(lang_hint: str) -> str:
        if not lang_hint or lang_hint.lower() == "auto":
            return "the original transcript language"
        return lang_hint


_NAT_RE = re.compile(r"(\d+)")


def _natural_sort_key(path: Path) -> tuple:
    """Case-insensitive natural sort key so files process alphabetically for RU/EN names and numbers."""
    name = path.name.casefold()
    parts = _NAT_RE.split(name)
    key: List[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    # include parent path for deterministic ordering across folders
    return (tuple(path.parts[:-1]), tuple(key))


def _derive_exclude_dirs(root: Path, output_dir: Optional[Path]) -> List[Path]:
    if output_dir is None:
        return []
    excludes: List[Path] = []
    root_resolved = root.resolve()
    output_resolved = Path(output_dir).expanduser().resolve()
    candidates: List[Path] = []
    if output_resolved != root_resolved:
        candidates.append(output_resolved)
    for name in _TOOL_OUTPUT_FOLDERS:
        candidates.append(output_resolved / name)
    for candidate in candidates:
        if _is_within(candidate, root_resolved):
            excludes.append(candidate)
    return excludes


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _is_under_any(path: Path, parents: Sequence[Path]) -> bool:
    for candidate in parents:
        if _is_within(path, candidate):
            return True
    return False
