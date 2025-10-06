"""ASR transcription utilities with VRAM-aware loading."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from utils.memory import VRAMLimitedModel, free_cuda, try_gpu_then_cpu

try:  # pragma: no cover - optional dependency during tests
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover
    WhisperModel = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ASRConfig:
    model: str
    device: str = "cuda"
    compute_type: str = "float16"
    max_parallel: int = 1
    vad: Optional[str] = None
    gpu_memory_limit: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ASRConfig":
        return cls(
            model=data.get("model") or data.get("name") or data.get("path") or "small",
            device=data.get("device", "cuda"),
            compute_type=data.get("compute_type", "float16"),
            max_parallel=int(data.get("max_parallel", 1)),
            vad=data.get("vad"),
            gpu_memory_limit=data.get("gpu_memory_limit") or data.get("gpu_memory") or int(7 * 1024),
        )


class ASRTranscriber:
    """ASR pipeline that loads whisper checkpoints lazily."""

    def __init__(self, config: Dict[str, Any], *, allow_remote: bool = False) -> None:
        self.config = ASRConfig.from_dict(config)
        self.allow_remote = allow_remote
        if WhisperModel is None:
            raise RuntimeError("faster-whisper is not installed")
        self._model_ctx: Optional[VRAMLimitedModel] = None
        self._model: Optional[WhisperModel] = None  # type: ignore[name-defined]
        self._current_device: Optional[str] = None

    def __enter__(self) -> "ASRTranscriber":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._release_model()
        return False

    # Public API -----------------------------------------------------------------
    def transcribe(self, media_path: Path, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def run_on_gpu() -> None:
            if self.config.device == "cpu":
                return run_on_cpu()
            model = self._ensure_model("cuda")
            self._transcribe_with_model(model, media_path, output_path)

        def run_on_cpu() -> None:
            model = self._ensure_model("cpu")
            self._transcribe_with_model(model, media_path, output_path)

        try_gpu_then_cpu(run_on_gpu, on_cpu=run_on_cpu if self.config.device != "cpu" else None)

    # Internal helpers ------------------------------------------------------------
    def _transcribe_with_model(
        self,
        model: "WhisperModel",
        media_path: Path,
        output_path: Path,
    ) -> None:
        segments, info = model.transcribe(
            str(media_path),
            beam_size=1,
            vad_filter=self.config.vad == "silero",
        )
        logger.info(
            "Transcribed %s (language=%s, duration=%.2fs)",
            media_path.name,
            getattr(info, "language", "unknown"),
            getattr(info, "duration", 0.0),
        )
        with output_path.open("w", encoding="utf-8") as f:
            for seg in segments:
                payload = {
                    "start": getattr(seg, "start", 0.0),
                    "end": getattr(seg, "end", 0.0),
                    "text": getattr(seg, "text", ""),
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _ensure_model(self, device: str) -> "WhisperModel":
        if self._model is not None and self._current_device == device:
            return self._model
        self._release_model()
        self._current_device = device
        gpu_limit = self.config.gpu_memory_limit
        if device == "cpu":
            gpu_limit = None
        logger.debug("Loading ASR model on %s with limit=%s", device, gpu_limit)

        def load() -> "WhisperModel":
            return WhisperModel(  # type: ignore[call-arg]
                self.config.model,
                device=device,
                compute_type=self.config.compute_type,
                local_files_only=not self.allow_remote,
                gpu_memory_limit=gpu_limit,
            )

        self._model_ctx = VRAMLimitedModel(load, self._unload_model)
        self._model = self._model_ctx.__enter__()
        return self._model

    def _release_model(self) -> None:
        if self._model_ctx is not None:
            self._model_ctx.__exit__(None, None, None)
        self._model_ctx = None
        self._model = None
        self._current_device = None
        free_cuda()

    @staticmethod
    def _unload_model(model: "WhisperModel") -> None:  # pragma: no cover - library handles cleanup
        try:
            model = None
        finally:
            free_cuda()
