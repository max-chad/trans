"""Local text correction utilities supporting LLM and rule-based backends."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from utils.memory import VRAMLimitedModel, free_cuda, try_gpu_then_cpu

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional heavy dependency
    from llama_cpp import Llama
except ImportError:  # pragma: no cover
    Llama = None  # type: ignore


class LocalTextCorrector:
    """Lazy-loading corrector that supports LLM and rule-based strategies."""

    def __init__(self, config: Dict[str, Any], *, allow_remote: bool = False) -> None:
        self.config = config
        self.allow_remote = allow_remote
        self.backend = config.get("backend", "rules").lower()
        self.model_path = config.get("model_path")
        self.quant = config.get("quant")
        self.max_input_len = int(config.get("max_input_len", 4096))
        self.batch_size = int(config.get("batch_size", config.get("max_batch", 4)))
        self._model_ctx: Optional[VRAMLimitedModel] = None
        self._model = None
        self._device: Optional[str] = None

    # ------------------------------------------------------------------
    def __enter__(self) -> "LocalTextCorrector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._release_model()
        return False

    # ------------------------------------------------------------------
    def correct_batch(self, texts: Iterable[str]) -> List[str]:
        batch = list(texts)
        if not batch:
            return []
        if self.backend in {"rules", "rule", "regex"}:
            return [self._rule_based_normalize(text) for text in batch]
        if self.backend in {"llm", "llama", "instructional"}:
            return self._llm_correct(batch)
        raise ValueError(f"Unsupported backend: {self.backend}")

    def correct(self, text: str) -> str:
        return self.correct_batch([text])[0]

    # LLM backend -------------------------------------------------------
    def _llm_correct(self, batch: List[str]) -> List[str]:
        if Llama is None:
            raise RuntimeError("llama-cpp-python is required for LLM backend")

        def run_on_gpu() -> List[str]:
            device = "cuda"
            if not self._gpu_available():
                return run_on_cpu()
            model = self._ensure_llm(device)
            return [self._run_prompt(model, item) for item in batch]

        def run_on_cpu() -> List[str]:
            model = self._ensure_llm("cpu")
            return [self._run_prompt(model, item) for item in batch]

        return try_gpu_then_cpu(run_on_gpu, on_cpu=run_on_cpu)

    def _run_prompt(self, model: Any, text: str) -> str:
        prompt = self._build_prompt(text)
        result = model(
            prompt,
            max_tokens=self.max_input_len,
            temperature=0.0,
            top_p=0.95,
        )
        choices = result.get("choices", [])
        if not choices:
            return text
        output = choices[0].get("text", "").strip()
        return output or text

    def _build_prompt(self, text: str) -> str:
        system_prompt = self.config.get(
            "system_prompt",
            "You are a transcription editor. Fix punctuation, casing, and typos while preserving meaning.",
        )
        return f"<|system|>\n{system_prompt}\n<|user|>\n{text.strip()}\n<|assistant|>"

    def _ensure_llm(self, device: str) -> Any:
        if self._model is not None and self._device == device:
            return self._model
        self._release_model()
        logger.debug("Loading LLM corrector on %s", device)

        def load() -> Any:
            params = {
                "model_path": self._resolve_model_path(),
                "n_ctx": self.max_input_len,
                "n_gpu_layers": -1 if device != "cpu" else 0,
                "seed": 0,
                "chat_format": self.config.get("chat_format", "chatml"),
            }
            if device == "cpu":
                params["n_gpu_layers"] = 0
            return Llama(**params)

        self._model_ctx = VRAMLimitedModel(load, self._unload_model)
        self._model = self._model_ctx.__enter__()
        self._device = device
        return self._model

    def _release_model(self) -> None:
        if self._model_ctx is not None:
            self._model_ctx.__exit__(None, None, None)
        self._model_ctx = None
        self._model = None
        self._device = None
        free_cuda()

    def _resolve_model_path(self) -> str:
        if not self.model_path:
            raise ValueError("llm.model_path must be specified for LLM backend")
        return str(Path(self.model_path).expanduser())

    @staticmethod
    def _gpu_available() -> bool:
        try:
            import torch  # type: ignore

            return torch.cuda.is_available()
        except Exception:  # pragma: no cover - torch may not be installed
            return False

    @staticmethod
    def _unload_model(model: Any) -> None:  # pragma: no cover - library handles cleanup
        del model
        free_cuda()

    # Rule backend ------------------------------------------------------
    _whitespace_re = re.compile(r"\s+")
    _punctuation_replacements = {
        r" ?!": "!",
        r" ?,": ",",
        r" ?\.": ".",
    }

    def _rule_based_normalize(self, text: str) -> str:
        original = text
        text = text.strip()
        text = self._whitespace_re.sub(" ", text)
        for pattern, replacement in self._punctuation_replacements.items():
            text = re.sub(pattern, replacement, text)
        text = re.sub(r"\s+([,.!?])", r"\1", text)
        text = re.sub(r"([!?]){2,}", r"\1", text)
        text = text[: self.max_input_len]
        logger.debug("Rule-based normalization: '%s' -> '%s'", original, text)
        return text
