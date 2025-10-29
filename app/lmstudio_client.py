import logging
import re
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

import requests
from requests import Response

# Logger keeps module-level diagnostics consistent with rest of app.
logger = logging.getLogger(__name__)

# System prompts define behavior for different subtitle workflows.
REWRITE_SYSTEM_PROMPT = (
    "You are an expert subtitle editor. Clean up grammar, spelling, and punctuation without changing meaning, tone, slang, or line breaks. "
    "Return exactly the same number of lines in the same order. Do not merge or split lines, add numbering, or include explanations."
)

REWRITE_DEEP_SYSTEM_PROMPT = (
    "You are a professional dialogue editor. Rephrase colloquial or fragmented speech into clear, natural sentences while preserving intent and key facts. "
    "Feel free to expand shorthand, clarify implied words, and smooth over filler phrases. Keep the number of lines and their order unchanged."
)

TRANSLATE_SYSTEM_PROMPT = (
    "You are a professional audiovisual translator. Translate the text while preserving context, tone, and approximate pacing. "
    "Return the same number of lines as the input, plain text only, without extra commentary."
)

# Regex strips common numbering prefixes that models might add.
_LINE_NUMBER_RE = re.compile(r"^\s*(?:\d+[\).:-]|\-\s+)\s*")


class LmStudioError(RuntimeError):
    """Raised when LM Studio request fails."""


@dataclass
class LmStudioSettings:
    base_url: str
    model: str
    api_key: str = ""
    timeout: float = 60.0
    temperature: float = 0.1
    max_completion_tokens: int = 4096
    max_prompt_tokens: int = 8192
    prompt_token_margin: int = 512
    preset_id: str = "1"


class LmStudioClient:
    def __init__(self, settings: LmStudioSettings):
        # Validate required settings eagerly so callers get immediate feedback.
        if not settings.base_url:
            raise ValueError("LM Studio base URL is empty.")
        if not settings.model:
            raise ValueError("LM Studio model is not specified.")
        self.settings = settings
        # Keep a dedicated session to reuse connections and shared headers.
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if settings.api_key:
            self.session.headers.update({"Authorization": f"Bearer {settings.api_key}"})
        raw_root = settings.base_url.rstrip("/")
        preferred_roots: List[str] = []
        if raw_root:
            if raw_root.endswith("/v1"):
                preferred_roots.append(raw_root)
                preferred_roots.append(raw_root[: -len("/v1")])
            else:
                preferred_roots.append(raw_root + "/v1")
                preferred_roots.append(raw_root)
        self._api_roots: List[str] = []
        for root in preferred_roots:
            normalized = root.rstrip("/")
            if normalized and normalized not in self._api_roots:
                self._api_roots.append(normalized)
        if not self._api_roots:
            raise ValueError("LM Studio base URL is invalid.")
        # Track variants of each REST endpoint so we can fall back between LM Studio versions.
        self._chat_endpoints = [root + "/chat/completions" for root in self._api_roots]
        self._model_endpoints = [root + "/models" for root in self._api_roots]
        self._load_endpoints = []
        for root in self._api_roots:
            self._load_endpoints.extend(
                [
                    root + "/models",
                    root + "/models/load",
                    root + "/internal/model/load",
                ]
            )
        # Cache working and failing endpoints so we can avoid redundant retries.
        self._active_chat_endpoint: Optional[str] = None
        self._invalid_chat_endpoints: set[str] = set()
        self._invalid_model_endpoints: set[str] = set()

    def _post(self, payload: dict) -> Response:
        # Prioritize the most recent working endpoint, then fall back to others.
        candidates = []
        if self._active_chat_endpoint:
            candidates.append(self._active_chat_endpoint)
        candidates.extend(
            ep for ep in self._chat_endpoints if ep not in candidates and ep not in self._invalid_chat_endpoints
        )
        last_error: Optional[Exception] = None
        for endpoint in candidates:
            try:
                response = self.session.post(
                    endpoint,
                    json=payload,
                    timeout=self.settings.timeout,
                )
                response.raise_for_status()
                data = None
                try:
                    data = response.json()
                except ValueError:
                    data = None
                if isinstance(data, dict) and "error" in data:
                    # Some endpoints reply with 200 but include an error payload; mark them as unusable.
                    self._invalid_chat_endpoints.add(endpoint)
                    last_error = LmStudioError(data.get("error") or "LM Studio returned error response.")
                    continue
                # Remember the successful endpoint to optimize subsequent calls.
                self._active_chat_endpoint = endpoint
                return response
            except requests.RequestException as exc:
                last_error = exc
                continue
        if last_error is None:
            raise LmStudioError("LM Studio chat endpoint is not configured.")
        raise LmStudioError(f"LM Studio request failed: {last_error}") from last_error

    def chat_completion(self, messages: List[dict], max_tokens: Optional[int] = None) -> str:
        # Compose a chat completion request while honoring optional token limits.
        budget = max_tokens if max_tokens is not None else self.settings.max_completion_tokens
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "stream": False,
        }
        if budget and budget > 0:
            payload["max_tokens"] = int(budget)
        response = self._post(payload)
        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - invalid json
            raise LmStudioError(f"Invalid LM Studio response: {response.text[:200]}") from exc
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            return str(content).strip()
        except (KeyError, IndexError, TypeError) as exc:
            # LM Studio occasionally responds with partial data; surface that upstream.
            raise LmStudioError(f"Incomplete LM Studio response: {data}") from exc

    def rewrite_batch(self, lines: List[str], lang_hint: str = "", mode: str = "polish") -> List[str]:
        if not lines:
            return []
        # Pick the rewrite style and instruction set based on caller hints.
        system_prompt = REWRITE_SYSTEM_PROMPT
        description = (
            f"Rewrite the following {len(lines)} subtitle lines without changing their order. "
            "Keep abbreviations and slang if present."
        )
        if mode.lower() == "deep":
            system_prompt = REWRITE_DEEP_SYSTEM_PROMPT
            description = (
                f"Rewrite the following {len(lines)} subtitle lines into polished narrative sentences. "
                "You may paraphrase heavily, clarify implied context, and expand casual speech while keeping the same number of lines."
            )
        if lang_hint:
            system_prompt += f" Source language hint: {lang_hint}."
        # Flatten multi-line entries so we keep the input shape for validation later.
        normalized = [line.replace("\n", " ").strip() for line in lines]
        messages = [
            {
                "role": "user",
                "content": _compose_user_prompt(system_prompt, description, normalized),
            }
        ]
        completion_budget = _completion_budget(normalized, self.settings.max_completion_tokens)
        raw = self.chat_completion(messages, max_tokens=completion_budget)
        parsed = _split_lines(raw, len(lines))
        if parsed is None:
            logger.warning(
                "LM Studio rewrite returned %s lines instead of %s. Keeping originals.",
                len([l for l in raw.splitlines() if l.strip()]),
                len(lines),
            )
            return lines
        return parsed

    def translate_batch(
        self,
        lines: List[str],
        target_lang: str,
        source_lang: Optional[str] = None,
    ) -> List[str]:
        if not lines:
            return []
        # Prepare translation instructions and normalize whitespace for better model input.
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

    def ensure_model_loaded(
        self,
        log_callback: Optional[Callable[[str, str], None]] = None,
        timeout: float = 120.0,
        poll_interval: float = 1.5,
    ) -> None:
        """Ensure target model is loaded in LM Studio, requesting load if needed."""
        # Wrap logging so callers can surface status updates in their own UI.
        def _log(level: str, message: str) -> None:
            if log_callback:
                try:
                    log_callback(level, message)
                except Exception:  # pragma: no cover - defensive
                    logger.exception("Failed to emit LM Studio log callback.")

        # Check whether the target is already active before touching the REST API.
        entry = self._find_model_entry()
        if self._is_model_ready(entry):
            return

        _log("info", f"Загрузка модели LM Studio: {self.settings.model}")
        if not self._request_model_load():
            raise LmStudioError("Не удалось инициировать загрузку модели в LM Studio.")

        # Poll until the service reports the model as ready or the timeout expires.
        deadline = time.time() + max(5.0, timeout)
        while time.time() < deadline:
            time.sleep(max(0.5, poll_interval))
            entry = self._find_model_entry()
            if self._is_model_ready(entry):
                _log("success", f"Модель LM Studio готова: {self.settings.model}")
                return
        raise LmStudioError("LM Studio не загрузила модель в отведённое время.")

    def _request_model_load(self) -> bool:
        preset = str(self.settings.preset_id).strip() if self.settings.preset_id is not None else ""
        if not preset:
            preset = "1"
        payload = {"model": self.settings.model, "presetId": preset}
        # Try each known load endpoint variant until one accepts the request.
        for endpoint in self._load_endpoints:
            try:
                response = self.session.post(endpoint, json=payload, timeout=self.settings.timeout)
            except requests.RequestException:
                continue
            if response.status_code in (200, 201, 202, 204):
                return True
            if response.status_code == 409:
                # Already loading/loaded
                return True
        return False

    def _find_model_entry(self) -> Optional[dict]:
        # Query each metadata endpoint and normalize the many response shapes LM Studio returns.
        for endpoint in self._model_endpoints:
            if endpoint in self._invalid_model_endpoints:
                continue
            try:
                response = self.session.get(endpoint, timeout=self.settings.timeout)
                response.raise_for_status()
            except requests.RequestException:
                continue
            try:
                data = response.json()
            except ValueError:
                continue
            if isinstance(data, dict) and "error" in data:
                # LM Studio returns 200 with error payload for unsupported endpoints. Skip them.
                self._invalid_model_endpoints.add(endpoint)
                continue
            entries: List[dict] = []
            if isinstance(data, dict):
                if isinstance(data.get("data"), list):
                    entries = [item for item in data["data"] if isinstance(item, dict)]
                elif isinstance(data.get("models"), list):
                    entries = [item for item in data["models"] if isinstance(item, dict)]
            elif isinstance(data, list):
                entries = [item for item in data if isinstance(item, dict)]
            target = self.settings.model.lower()
            for entry in entries:
                identifier = str(entry.get("id") or entry.get("model") or "").lower()
                if identifier == target:
                    return entry
        return None

    @staticmethod
    def _is_model_ready(entry: Optional[dict]) -> bool:
        # LM Studio uses multiple flags across builds; accept any that imply readiness.
        if not entry:
            return False
        status = str(entry.get("status") or entry.get("state") or "").lower()
        if status in {"ready", "loaded", "active", "online"}:
            return True
        if entry.get("ready") is True or entry.get("loaded") is True:
            return True
        if entry.get("isLoaded") is True or entry.get("loadedOnGPU") is True:
            return True
        inference = entry.get("inferenceStatus") or entry.get("inference_status")
        if isinstance(inference, str) and inference.lower() in {"ready", "active"}:
            return True
        # Some LM Studio endpoints return minimal metadata without status field.
        # If we see the model entry at all and no explicit "loading" indicator, assume ready.
        if status == "" and inference in (None, ""):
            return True
        return False


def _split_lines(raw: str, expected: int) -> Optional[List[str]]:
    # Attempt to coerce the model output back into the original line count.
    if not raw:
        return None
    candidates = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(candidates) == expected:
        return candidates
    cleaned = [_LINE_NUMBER_RE.sub("", line).strip() for line in candidates]
    # Many models add numbering when rewriting; strip it and try again.
    cleaned = [line for line in cleaned if line]
    if len(cleaned) == expected:
        return cleaned
    if expected == 1 and raw.strip():
        return [raw.strip()]
    return None


def _estimate_tokens(text: str) -> int:
    # Cheap heuristic that keeps budgets reasonable without external tokenizer deps.
    stripped = text.strip()
    if not stripped:
        return 1
    by_chars = (len(stripped) + 3) // 4
    by_words = int(len(stripped.split()) * 1.1) or 1
    return max(1, by_chars, by_words)


def _completion_budget(lines: List[str], default: int, multiplier: float = 1.4) -> int:
    # Scale completion allowance to input size but cap it to the configured defaults.
    estimated = sum(_estimate_tokens(line) for line in lines)
    target = int(estimated * multiplier) + 64
    if default and default > 0:
        return min(default, max(128, target))
    return max(128, target)


def _compose_user_prompt(instructions: str, description: str, lines: List[str]) -> str:
    # Merge system-level instructions into the user message for models that reject system roles.
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
    # Filter leading/trailing blanks introduced by missing sections.
    return "\n".join(part for part in sections if part is not None)


def chunked(
    iterable: List[str],
    size: int,
    max_tokens: int = 0,
    token_margin: int = 512,
) -> Iterable[List[str]]:
    # Yield subtitle batches that respect both size and approximate token limits.
    if not iterable:
        return
    max_size = max(1, size)
    # Reserve part of the token budget for system prompts and glue text.
    effective_limit = max(0, max_tokens - max(0, token_margin)) if max_tokens else 0
    chunk: List[str] = []
    token_count = 0
    for item in iterable:
        text = item if isinstance(item, str) else str(item)
        tokens = _estimate_tokens(text)
        if effective_limit and tokens > effective_limit:
            # Emit oversized entries on their own so we do not drop content.
            if chunk:
                yield chunk
                chunk = []
                token_count = 0
            yield [text]
            continue
        next_token_total = token_count + tokens
        if chunk and (len(chunk) >= max_size or (effective_limit and next_token_total > effective_limit)):
            # Flush the current chunk when we hit either item or token thresholds.
            yield chunk
            chunk = []
            token_count = 0
        chunk.append(text)
        token_count += tokens
    if chunk:
        yield chunk
