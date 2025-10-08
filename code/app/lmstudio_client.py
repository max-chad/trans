import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

import requests
from requests import Response

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = (
    "Ты — точный корректор субтитров. Исправляй ошибки распознавания по контексту, "
    "не меняй смысл и стиль. Сохраняй язык исходных строк, их количество и порядок."
)

TRANSLATE_SYSTEM_PROMPT = (
    "Ты — профессиональный переводчик субтитров. Переводи строки точно и кратко, "
    "сохраняй порядок и количество строк, не добавляй новые элементы."
)


class LmStudioError(RuntimeError):
    """Raised when LM Studio request fails."""


@dataclass
class LmStudioSettings:
    base_url: str
    model: str
    api_key: str = ""
    timeout: float = 60.0
    temperature: float = 0.1


class LmStudioClient:
    def __init__(self, settings: LmStudioSettings):
        if not settings.base_url:
            raise ValueError("LM Studio base URL is empty.")
        if not settings.model:
            raise ValueError("LM Studio model is not specified.")
        self.settings = settings
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if settings.api_key:
            self.session.headers.update({"Authorization": f"Bearer {settings.api_key}"})
        self._api_chat = settings.base_url.rstrip("/") + "/chat/completions"

    def _post(self, payload: dict) -> Response:
        try:
            response = self.session.post(
                self._api_chat,
                json=payload,
                timeout=self.settings.timeout,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            raise LmStudioError(f"LM Studio request failed: {exc}") from exc

    def chat_completion(self, messages: List[dict], max_tokens: int = 1024) -> str:
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "stream": False,
            "max_tokens": max_tokens,
        }
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
            raise LmStudioError(f"Incomplete LM Studio response: {data}") from exc

    def rewrite_batch(self, lines: List[str], lang_hint: str = "") -> List[str]:
        if not lines:
            return []
        system_prompt = REWRITE_SYSTEM_PROMPT
        if lang_hint:
            system_prompt += f" Язык текста: {lang_hint}."
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Всего {len(lines)} строк. Исправь каждую, сохрани последовательность.\n"
                    + "\n".join(line.replace("\n", " ").strip() for line in lines)
                ),
            },
        ]
        raw = self.chat_completion(messages)
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
        description = (
            f"Переведи {len(lines)} строк на {target_lang}. "
            "Сохраняй порядок и количество строк."
        )
        if source_lang and source_lang.lower() != "auto":
            description += f" Исходный язык: {source_lang}."
        messages = [
            {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": description + "\n" + "\n".join(line.replace("\n", " ").strip() for line in lines),
            },
        ]
        raw = self.chat_completion(messages)
        parsed = _split_lines(raw, len(lines))
        if parsed is None:
            logger.warning(
                "LM Studio translate returned %s lines instead of %s. Keeping originals.",
                len([l for l in raw.splitlines() if l.strip()]),
                len(lines),
            )
            return lines
        return parsed


def _split_lines(raw: str, expected: int) -> Optional[List[str]]:
    if not raw:
        return None
    candidates = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(candidates) == expected:
        return candidates
    return None


def chunked(iterable: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]
