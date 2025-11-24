from __future__ import annotations

from functools import lru_cache
from typing import Callable, List, Optional

try:  # pragma: no cover - optional dependency guards
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None

TokenCounter = Callable[[str], int]
_TOKEN_COUNTER: Optional[TokenCounter] = None
_FORCE_FALLBACK_MODELS = set()
_MODEL_ENCODING_ALIASES: dict[str, str] = {
    # OpenRouter Polaris shares the OpenAI o200k tokenizer (256k context window).
    "openrouter/polaris-alpha": "o200k_base",
    "polaris-alpha": "o200k_base",
}


def build_token_counter(model_hint: Optional[str]) -> TokenCounter:
    """
    Return a callable that estimates tokens for text using tiktoken when available.
    Falls back to a lightweight heuristic if encodings cannot be resolved.
    """

    if tiktoken is None:
        return _fallback_counter

    encoding = _resolve_encoding(model_hint)
    if encoding is None:
        return _fallback_counter

    def _count(text: str) -> int:
        if not text:
            return 1
        # Using encode_ordinary avoids reserving slots for special tokens.
        try:
            return max(1, len(encoding.encode_ordinary(text)))
        except Exception:
            # Some community models can trigger unexpected errors; do not crash.
            return _fallback_counter(text)

    return _count


def configure_token_counter(counter: Optional[TokenCounter]) -> None:
    global _TOKEN_COUNTER
    _TOKEN_COUNTER = counter


def estimate_tokens(text: str) -> int:
    """Return the configured token count or a heuristic fallback."""
    if _TOKEN_COUNTER:
        try:
            return max(1, _TOKEN_COUNTER(text))
        except Exception:
            pass
    return _fallback_counter(text)


@lru_cache(maxsize=8)
def _resolve_encoding(model_hint: Optional[str]):
    if tiktoken is None:
        return None
    candidates = _model_hint_candidates(model_hint)
    if any(candidate in _FORCE_FALLBACK_MODELS for candidate in candidates):
        return None
    for candidate in candidates:
        alias = _MODEL_ENCODING_ALIASES.get(candidate)
        if alias:
            try:
                return tiktoken.get_encoding(alias)
            except Exception:
                continue
        if not candidate:
            continue
        try:
            return tiktoken.encoding_for_model(candidate)
        except KeyError:
            continue
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _fallback_counter(text: str) -> int:
    if not text:
        return 1
    ascii_chars = 0
    non_ascii_chars = 0
    for char in text:
        if char.isascii():
            ascii_chars += 1
        else:
            non_ascii_chars += 1
    ascii_estimate = (ascii_chars + 3) // 4 if ascii_chars else 0
    non_ascii_estimate = non_ascii_chars + max(1, non_ascii_chars // 5) if non_ascii_chars else 0
    estimated = ascii_estimate + non_ascii_estimate
    return max(1, estimated)


def _model_hint_candidates(model_hint: Optional[str]) -> List[str]:
    if not model_hint:
        return [""]
    raw = model_hint.strip().lower()
    if not raw:
        return [""]
    candidates = [raw]
    if ":" in raw:
        prefix = raw.split(":", 1)[0]
        if prefix not in candidates:
            candidates.append(prefix)
    if "/" in raw:
        suffix = raw.rsplit("/", 1)[-1]
        if suffix not in candidates:
            candidates.append(suffix)
    return candidates
