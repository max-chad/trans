from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import requests
from requests import Response, Session
from requests.exceptions import RequestException

from app import token_utils

logger = logging.getLogger(__name__)

_LINE_NUMBER_RE = re.compile(r"^\s*(?:\d+[).:-]|\-\s+)\s*")
TRANSLATE_SYSTEM_PROMPT = (
    "You are a professional audiovisual translator. Translate the text while preserving context, tone, and approximate pacing. "
    "Return the same number of lines as the input, plain text only, without extra commentary."
)


@dataclass(frozen=True)
class ModelCapabilities:
    """Minimal metadata needed to decide whether a model exposes reasoning controls."""

    model_id: Optional[str]
    supported_parameters: Tuple[str, ...]
    supports_reasoning: bool
    reasoning_levels: Tuple[str, ...]


class LmStudioError(RuntimeError):
    """Raised when LM Studio returns an error or cannot be reached."""

    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


_DEFAULT_REASONING_LEVELS = ("low", "medium", "high")
REASONING_DEFAULT_SELECTION = "default"

@dataclass(slots=True)
class LmStudioSettings:
    base_url: str
    model: str
    api_key: str = ""
    timeout: float = 120.0
    max_completion_tokens: Optional[int] = None
    max_prompt_tokens: Optional[int] = None
    preset_id: str = "1"
    temperature: float = 0.2
    top_p: float = 0.9
    prompt_token_margin: int = 0
    provider_preferences: Optional[Dict[str, Any]] = None
    reasoning_effort: Optional[str] = None


class LmStudioClient:
    def __init__(self, settings: LmStudioSettings, session: Optional[Session] = None):
        if not settings.base_url:
            raise ValueError("LM Studio base URL is required.")
        if not settings.model:
            raise ValueError("LM Studio model is required.")
        self.settings = settings
        self._session = session or requests.Session()
        self._max_attempts = 3

    @property
    def session(self) -> Session:
        return self._session

    @session.setter
    def session(self, value: Session) -> None:
        self._session = value

    def ensure_model_loaded(
        self,
        log_callback: Callable[[str, str], None],
        timeout: float = 600.0,
        interval: Optional[float] = None,
        *,
        poll_interval: Optional[float] = None,
    ) -> None:
        deadline = time.monotonic() + timeout
        log_callback("info", f"Ensuring LM Studio model '{self.settings.model}' is loaded.")
        ping_payload = {
            "model": self.settings.model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
        if self.settings.preset_id:
            ping_payload["presetId"] = str(self.settings.preset_id)
        ping_payload = self._apply_provider_preferences(ping_payload)
        sleep_interval = poll_interval if poll_interval is not None else interval if interval is not None else 5.0
        while True:
            try:
                self._post("chat/completions", ping_payload)
                log_callback("success", "LM Studio is ready.")
                return
            except LmStudioError as exc:
                if not getattr(exc, "retryable", True):
                    log_callback("error", f"LM Studio returned a fatal error: {exc}")
                    raise
                if time.monotonic() >= deadline:
                    raise
                log_callback("warning", f"LM Studio not ready: {exc}. Retrying...")
                time.sleep(sleep_interval)

    def chat_completion(self, messages: Sequence[dict], max_tokens: Optional[int] = None) -> str:
        payload = {
            "model": self.settings.model,
            "messages": list(messages),
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
        }
        requested_limit = max_tokens if (max_tokens or 0) > 0 else self.settings.max_completion_tokens
        if requested_limit and requested_limit > 0:
            if (
                max_tokens
                and self.settings.max_completion_tokens
                and self.settings.max_completion_tokens > 0
            ):
                requested_limit = min(requested_limit, self.settings.max_completion_tokens)
            payload["max_tokens"] = int(requested_limit)
        payload = self._apply_reasoning_config(payload)
        payload = self._apply_provider_preferences(payload)
        data = self._post("chat/completions", payload)
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            return content.strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LmStudioError(f"Unexpected response format: {data}") from exc

    def rewrite_batch(self, lines: Sequence[str], language_hint: str, mode: str) -> List[str]:
        instruction = _mode_instruction(mode)
        system_prompt = (
            "You are a professional subtitle editor. Rewrite each line individually and preserve the order. "
            "Return a JSON array of strings with the same length as the input."
        )
        user_payload = {
            "language_hint": language_hint or "auto",
            "mode": instruction,
            "lines": list(lines),
        }
        response = self.chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            max_tokens=self.settings.max_completion_tokens,
        )
        try:
            parsed = json.loads(response)
            if isinstance(parsed, list):
                normalized = [str(item).strip() for item in parsed]
                return _align_lines(normalized, len(lines), list(lines))
        except json.JSONDecodeError:
            pass
        fallback = [line.strip() for line in response.splitlines() if line.strip()]
        return _align_lines(fallback, len(lines), list(lines))

    def translate_batch(
        self,
        lines: List[str],
        target_lang: str,
        source_lang: Optional[str] = None,
    ) -> List[str]:
        if not lines:
            return []
        normalized = [line.replace("\n", " ").strip() for line in lines]
        description = (
            f"Translate the following {len(normalized)} subtitle lines into {target_lang}. "
            "Preserve meaning, tone, and the length of each line."
        )
        if source_lang and source_lang.lower() != "auto":
            description += f" The original language is {source_lang}."
        messages = [
            {
                "role": "user",
                "content": _compose_user_prompt(TRANSLATE_SYSTEM_PROMPT, description, normalized),
            }
        ]
        completion_budget = _completion_budget(normalized, self.settings.max_completion_tokens, multiplier=1.6)
        raw = self.chat_completion(messages, max_tokens=completion_budget)
        parsed = _split_lines(raw, len(lines))
        if parsed is None:
            logger.warning(
                "LM Studio translate returned %s lines instead of %s. Keeping originals.",
                len([l for l in raw.splitlines() if l.strip()]),
                len(lines),
            )
            return lines
        return parsed

    def _post(self, path: str, payload: dict) -> dict:
        url = self._url(path)
        attempt = 1
        last_error: Optional[LmStudioError] = None
        while attempt <= self._max_attempts:
            try:
                request_timeout = self.settings.timeout if self.settings.timeout is not None else 120.0
                response: Response = self._session.post(
                    url, json=payload, headers=self._headers(), timeout=request_timeout
                )
            except RequestException as exc:
                retryable = self._should_retry(attempt)
                error = LmStudioError(f"Failed to reach LM Studio: {exc}", retryable=retryable)
                if not retryable:
                    raise error from exc
                last_error = error
                self._sleep(self._retry_delay(attempt))
                attempt += 1
                continue

            if response.status_code >= 400:
                retryable = self._should_retry(attempt, response.status_code)
                error = LmStudioError(
                    f"LM Studio error {response.status_code}: {self._clip_text(response.text)}",
                    retryable=retryable,
                )
                if not retryable:
                    raise error
                last_error = error
                self._sleep(self._retry_delay(attempt))
                attempt += 1
                continue

            try:
                data = response.json()
            except ValueError:
                fallback = self._extract_json_payload(response.text)
                if fallback is not None:
                    data = fallback
                else:
                    retryable = self._should_retry(attempt)
                    error = LmStudioError(
                        f"Invalid JSON response from LM Studio: {self._clip_text(response.text)}",
                        retryable=retryable,
                    )
                    if not retryable:
                        raise error
                    last_error = error
                    self._sleep(self._retry_delay(attempt))
                    attempt += 1
                    continue

            provider_error = self._provider_error_payload(data)
            if provider_error:
                message, code = provider_error
                retryable = self._should_retry(attempt, code)
                error = LmStudioError(message, retryable=retryable)
                if not retryable:
                    raise error
                last_error = error
                self._sleep(self._retry_delay(attempt))
                attempt += 1
                continue

            return data
        if last_error:
            raise last_error
        raise LmStudioError("LM Studio request failed.")

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        if self.settings.preset_id:
            headers["X-LLM-Preset-ID"] = str(self.settings.preset_id)
        return headers

    def _url(self, path: str) -> str:
        base = self.settings.base_url.rstrip("/")
        return f"{base}/{path.lstrip('/')}"

    @staticmethod
    def _models_endpoint(base_url: str) -> str:
        base = base_url.rstrip("/")
        return f"{base}/models"

    @staticmethod
    def _matches_entry(entry: Dict[str, Any], normalized_target: str) -> bool:
        for key in ("id", "model", "canonical_slug", "model_id", "slug", "name"):
            candidate = entry.get(key)
            if isinstance(candidate, str) and candidate.strip().lower() == normalized_target:
                return True
        return False

    @staticmethod
    def fetch_model_capabilities(
        base_url: str,
        *,
        model: str | None = None,
        api_key: str = "",
        preset_id: Optional[str] = None,
        session: Optional[Session] = None,
        timeout: float = 20.0,
    ) -> ModelCapabilities:
        url = LmStudioClient._models_endpoint(base_url)
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if preset_id:
            headers["X-LLM-Preset-ID"] = str(preset_id)
        params = {"model": model} if model else None
        session = session or requests.Session()
        response = session.get(url, headers=headers, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        entries = None
        if isinstance(data, dict):
            entries = data.get("data")
            if entries is None:
                entries = data.get("models")
        if entries is None:
            if isinstance(data, list):
                entries = data
            else:
                entries = [data]
        if isinstance(entries, dict):
            entries = [entries]
        normalized_target = (model or "").strip().lower()
        chosen: Optional[Dict[str, Any]] = None
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if normalized_target and LmStudioClient._matches_entry(entry, normalized_target):
                chosen = entry
                break
            if chosen is None:
                chosen = entry
        supported_params: Tuple[str, ...] = ()
        if chosen:
            raw_params = chosen.get("supported_parameters") or []
            if isinstance(raw_params, list):
                supported_params = tuple(
                    param.strip().lower()
                    for param in raw_params
                    if isinstance(param, str) and param.strip()
                )
        supports_reasoning = "reasoning" in supported_params
        reasoning_levels = _DEFAULT_REASONING_LEVELS
        if chosen:
            candidate_levels = chosen.get("reasoning_levels") or chosen.get("reasoning_effort_levels")
            if isinstance(candidate_levels, (list, tuple)):
                normalized_levels = tuple(
                    level.strip().lower()
                    for level in candidate_levels
                    if isinstance(level, str) and level.strip()
                )
                if normalized_levels:
                    reasoning_levels = normalized_levels
        model_id = None
        if chosen:
            candidate_id = chosen.get("id") or chosen.get("canonical_slug") or chosen.get("model")
            if isinstance(candidate_id, str):
                model_id = candidate_id.strip()
        return ModelCapabilities(
            model_id=model_id,
            supported_parameters=supported_params,
            supports_reasoning=supports_reasoning,
            reasoning_levels=reasoning_levels,
        )

    @staticmethod
    def _clip_text(text: str, limit: int = 800) -> str:
        snippet = (text or "").strip()
        if not snippet:
            return "<empty response>"
        if len(snippet) <= limit:
            return snippet
        return f"{snippet[:limit]}…"

    @staticmethod
    def _retry_delay(attempt: int) -> float:
        return min(5.0, 0.5 * attempt)

    def _should_retry(self, attempt: int, status_code: Optional[int] = None) -> bool:
        if attempt >= self._max_attempts:
            return False
        if status_code is None:
            return True
        retryable = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 524}
        if status_code in retryable:
            return True
        if 500 <= status_code < 600:
            return True
        return False

    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    @staticmethod
    def _extract_json_payload(body: str) -> Optional[dict]:
        stripped = (body or "").strip()
        if not stripped:
            return None
        for candidate in LmStudioClient._json_candidates(stripped):
            try:
                loaded = json.loads(candidate)
            except ValueError:
                continue
            if isinstance(loaded, dict):
                return loaded
        sse_payload = LmStudioClient._parse_sse_chunks(stripped)
        if sse_payload:
            return sse_payload
        return None

    @staticmethod
    def _json_candidates(body: str) -> Iterator[str]:
        yield body
        if body.startswith("```"):
            inner = "\n".join(
                line for line in body.splitlines() if not line.strip().startswith("```")
            ).strip()
            if inner:
                yield inner
        first = body.find("{")
        last = body.rfind("}")
        if 0 <= first < last:
            yield body[first : last + 1]

    @staticmethod
    def _parse_sse_chunks(body: str) -> Optional[dict]:
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        if not any(line.startswith("data:") for line in lines):
            return None
        content_parts: List[str] = []
        finish_reason: Optional[str] = None
        role: Optional[str] = None
        for line in lines:
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except ValueError:
                continue
            if not isinstance(chunk, dict):
                continue
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            text_piece = delta.get("content")
            if text_piece:
                content_parts.append(text_piece)
            if not role and isinstance(delta.get("role"), str):
                role = delta["role"]
            finish_reason = choice.get("finish_reason") or finish_reason
        if not content_parts:
            return None
        content = "".join(content_parts)
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": role or "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ]
        }

    @staticmethod
    def _provider_error_payload(payload: object) -> Optional[tuple[str, Optional[int]]]:
        if not isinstance(payload, dict):
            return None
        error_block = payload.get("error")
        if not isinstance(error_block, dict):
            return None
        message = str(error_block.get("message") or "Unknown LM Studio error")
        metadata = error_block.get("metadata")
        details: list[str] = []
        if isinstance(metadata, dict):
            raw = metadata.get("raw")
            provider = metadata.get("provider_name")
            if provider:
                details.append(f"provider={provider}")
            if raw:
                details.append(raw)
        if details:
            message = f"{message} ({'; '.join(details)})"
        code_raw = error_block.get("code")
        code: Optional[int] = None
        if isinstance(code_raw, int):
            code = code_raw
        else:
            try:
                code = int(str(code_raw))
            except (TypeError, ValueError):
                code = None
        if code is not None:
            formatted = f"LM Studio error {code}: {message}"
        else:
            formatted = f"LM Studio error: {message}"
        return formatted, code

    def _apply_reasoning_config(self, payload: dict) -> dict:
        effort = (self.settings.reasoning_effort or "").strip()
        if not effort:
            return payload
        normalized = effort.lower()
        reasoning_block = dict(payload.get("reasoning") or {})
        if normalized != REASONING_DEFAULT_SELECTION:
            reasoning_block["effort"] = effort
        reasoning_block.setdefault("enabled", True)
        enriched = dict(payload)
        enriched["reasoning"] = reasoning_block
        return enriched

    def _apply_provider_preferences(self, payload: dict) -> dict:
        preferences = self.settings.provider_preferences or {}
        cleaned = self._clean_provider_preferences(preferences)
        if not cleaned:
            return payload
        enriched = dict(payload)
        enriched["provider"] = cleaned
        return enriched

    @staticmethod
    def _clean_provider_preferences(raw: Dict[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for key, value in raw.items():
            normalized = LmStudioClient._normalize_provider_value(value)
            if normalized is not None:
                cleaned[key] = normalized
        return cleaned

    @staticmethod
    def _normalize_provider_value(value: Any) -> Any | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        if isinstance(value, (list, tuple)):
            items = [
                normalized
                for normalized in (LmStudioClient._normalize_provider_value(item) for item in value)
                if normalized is not None
            ]
            return items or None
        if isinstance(value, dict):
            nested = {
                k: v
                for k, v in (
                    (inner_key, LmStudioClient._normalize_provider_value(inner_value))
                    for inner_key, inner_value in value.items()
                )
                if v is not None
            }
            return nested or None
        return value


def _is_local_url(url: str) -> bool:
    normalized = url or ""
    parsed = urlparse(normalized if "://" in normalized else f"http://{normalized}")
    host = (parsed.hostname or "").lower()
    return host in {"127.0.0.1", "localhost"} or host.startswith("192.168.") or host.startswith("10.")


def validate_lmstudio_settings(
    base_url: str,
    model: str,
    api_key: str = "",
    *,
    require_api_key: Optional[bool] = None,
) -> tuple[bool, str]:
    url = (base_url or "").strip()
    model_id = (model or "").strip()
    key = (api_key or "").strip()
    missing: list[str] = []
    if not url:
        missing.append("URL")
    if not model_id:
        missing.append("model")
    enforce_key = require_api_key if require_api_key is not None else bool(url and not _is_local_url(url))
    if enforce_key and not key:
        missing.append("API key")
    if missing:
        hint = ", ".join(missing)
        return False, f"LM Studio settings incomplete: {hint} required."
    return True, ""


def chunked(
    lines: Iterable[str],
    size: Optional[int] = None,
    max_tokens: Optional[int] = None,
    token_margin: Optional[int] = None,
    token_counter: Optional[Callable[[str], int]] = None,
) -> Iterator[List[str]]:
    bucket: List[str] = []
    bucket_tokens = 0
    limit = None
    counter = token_counter or estimate_tokens
    if max_tokens:
        safe_margin = max(0, token_margin or 0)
        limit = max(1, max_tokens - safe_margin)
    size_limit = max(1, size) if size and size > 0 else None
    for line in lines:
        line = line.rstrip("\n")
        token_cost = counter(line)
        exceeds_size = size_limit is not None and len(bucket) >= size_limit
        exceeds_tokens = limit is not None and (bucket_tokens + token_cost) > limit
        if bucket and (exceeds_size or exceeds_tokens):
            yield bucket
            bucket = []
            bucket_tokens = 0
        bucket.append(line)
        bucket_tokens += token_cost
    if bucket:
        yield bucket


estimate_tokens = token_utils.estimate_tokens
configure_token_counter = token_utils.configure_token_counter
# Backwards compatibility for older imports that referenced the underscored alias.
_estimate_tokens = estimate_tokens


def _mode_instruction(mode: str) -> str:
    normalized = (mode or "").lower()
    if normalized == "deep":
        return (
            "Deep rewrite: restructure aggressively while preserving factual meaning. "
            "Strip emotional markers, filler reactions, and tone indicators so the output stays neutral and "
            "dry. If an emotionally charged phrase or insult is essential to the message (e.g., it conveys a "
            "speaker's accusation or label), keep it verbatim."
        )
    return "Polish rewrite: keep timing and sentences but improve grammar and clarity."


def _align_lines(candidates: List[str], expected: int, fallback: List[str]) -> List[str]:
    if len(candidates) == expected:
        return candidates
    if len(candidates) > expected:
        return candidates[:expected]
    padded = list(candidates)
    for idx in range(len(padded), expected):
        padded.append(fallback[idx])
    return padded


def _split_lines(raw: str, expected: int) -> Optional[List[str]]:
    if not raw:
        return None
    candidates = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(candidates) == expected:
        return candidates
    cleaned = [_LINE_NUMBER_RE.sub("", line).strip() for line in candidates]
    cleaned = [line for line in cleaned if line]
    if len(cleaned) == expected:
        return cleaned
    if expected == 1 and raw.strip():
        return [raw.strip()]
    return None


def _completion_budget(lines: List[str], default: Optional[int], multiplier: float = 1.4) -> int:
    estimated = sum(token_utils.estimate_tokens(line) for line in lines)
    target = int(estimated * multiplier) + 64
    if default and default > 0:
        return min(default, max(128, target))
    return max(128, target)


def _compose_user_prompt(instructions: str, description: str, lines: List[str]) -> str:
    sections: List[str] = []
    if instructions:
        sections.append("Instructions:")
        sections.append(instructions.strip())
    if description:
        sections.append("")
        sections.append("Task:")
        sections.append(description.strip())
    if lines:
        sections.append("")
        sections.append("Input:")
        sections.append("\n".join(lines))
    return "\n".join(part for part in sections if part is not None)
