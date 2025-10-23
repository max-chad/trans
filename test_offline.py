"""
Utility checks for the offline pipeline.
Each test prints a short status and exits gracefully when prerequisites are missing.
Run as: python test_offline.py
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import math
import os
import tempfile
import wave
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List

import srt

import app.lmstudio_client as lmstudio_module
from app.lmstudio_client import LmStudioClient, LmStudioSettings

_local_llm_spec = importlib.util.find_spec("app.local_llm")
if _local_llm_spec is not None:
    _local_llm = importlib.import_module("app.local_llm")
    g4f_batch_rewrite = _local_llm.g4f_batch_rewrite  # type: ignore[attr-defined]
    g4f_batch_translate = _local_llm.g4f_batch_translate  # type: ignore[attr-defined]
    LOCAL_LLM_AVAILABLE = True
else:
    LOCAL_LLM_AVAILABLE = False

    def g4f_batch_rewrite(lines: List[str], model: str) -> List[str]:  # pragma: no cover - stub for type checkers
        raise RuntimeError("local LLM module not available")

    def g4f_batch_translate(
        lines: List[str],
        model: str,
        target_lang: str,
        source_lang: str,
    ) -> List[str]:  # pragma: no cover - stub for type checkers
        raise RuntimeError("local LLM module not available")

DEFAULT_SAMPLE = Path(__file__).with_name("tmp_test.wav")
ENV_SAMPLE_PATH = "TEST_WHISPER_SAMPLE"


def _generate_sine_wave(path: Path, seconds: int = 10, freq: int = 440, sample_rate: int = 16000):
    sample_count = seconds * sample_rate
    amplitude = 32767
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(sample_count):
            value = int(amplitude * math.sin(2 * math.pi * freq * (i / sample_rate)))
            wav.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))


def test_transcription_roundtrip():
    try:
        import whisper  # type: ignore
    except ImportError:
        print("SKIP test_transcription_roundtrip: whisper not installed")
        return
    model_name = os.getenv("TEST_WHISPER_MODEL", "tiny")
    try:
        model = whisper.load_model(model_name, device="cpu")
    except Exception as exc:  # pragma: no cover - depends on local setup
        print(f"SKIP test_transcription_roundtrip: cannot load model '{model_name}': {exc}")
        return
    assert model is not None, "Whisper model failed to load"
    sample_override = os.getenv(ENV_SAMPLE_PATH)
    candidate_sample = Path(sample_override).expanduser() if sample_override else DEFAULT_SAMPLE
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        if candidate_sample.exists():
            audio_path = candidate_sample
        else:
            audio_path = tmp_dir_path / "sample.wav"
            _generate_sine_wave(audio_path, seconds=10)
        result = model.transcribe(str(audio_path), language="en", verbose=False)
        segments_raw = result.get("segments")
        if not isinstance(segments_raw, list):
            sample_hint = f"sample '{audio_path}'"
            print(f"SKIP test_transcription_roundtrip: transcription produced no segments for {sample_hint}")
            return
        segments: List[Dict[str, Any]] = [seg for seg in segments_raw if isinstance(seg, dict)]
        if not segments:
            sample_hint = f"sample '{audio_path}'"
            print(f"SKIP test_transcription_roundtrip: transcription produced no usable segments for {sample_hint}")
            return
        txt_path = tmp_dir_path / "out.txt"
        srt_path = tmp_dir_path / "out.srt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(seg["text"].strip() + "\n")
        assert txt_path.exists(), "TXT export failed"
        srt_segments = [
            srt.Subtitle(
                index=i,
                start=timedelta(seconds=seg["start"]),
                end=timedelta(seconds=seg["end"]),
                content=seg["text"].strip(),
            )
            for i, seg in enumerate(segments, 1)
        ]
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt.compose(srt_segments))
        assert srt_path.exists(), "SRT export failed"
    print("OK test_transcription_roundtrip")


def test_rewrite_preserves_line_count():
    if not LOCAL_LLM_AVAILABLE:
        print("SKIP test_rewrite_preserves_line_count: local LLM module not available")
        return
    sample = ["Privet", "kak", "dela", "segodnya", "oshibka tyet"]
    corrected = g4f_batch_rewrite(sample, model="")
    assert len(corrected) == len(sample), "Rewrite changed line count"
    print("OK test_rewrite_preserves_line_count")


def test_translate_preserves_line_count():
    if not LOCAL_LLM_AVAILABLE:
        print("SKIP test_translate_preserves_line_count: local LLM module not available")
        return
    sample = [
        "\u041f\u0440\u0438\u0432\u0435\u0442",
        "\u041a\u0430\u043a \u0434\u0435\u043b\u0430?",
        "\u042d\u0442\u043e \u043e\u0444\u043b\u0430\u0439\u043d \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0430",
        "\u041f\u0440\u043e\u0432\u0435\u0440\u044f\u0435\u043c \u043f\u0435\u0440\u0435\u0432\u043e\u0434",
        "\u041e\u0434\u043d\u043e\u0439 \u0441\u0442\u0440\u043e\u043a\u043e\u0439",
    ]
    translated = g4f_batch_translate(sample, model="", target_lang="en", source_lang="ru")
    assert len(translated) == len(sample), "Translation changed line count"
    back = g4f_batch_translate(translated, model="", target_lang="ru", source_lang="en")
    assert len(back) == len(sample), "Reverse translation changed line count"
    print("OK test_translate_preserves_line_count")


def test_fail_safe_logging():
    if not LOCAL_LLM_AVAILABLE:
        print("SKIP test_fail_safe_logging: local LLM module not available")
        return
    logger = logging.getLogger("app.local_llm")
    captured: List[logging.LogRecord] = []

    class ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
            captured.append(record)

    handler = ListHandler()
    handler.setLevel(logging.WARNING)
    logger.addHandler(handler)
    previous_level = logger.level
    logger.setLevel(logging.WARNING)
    sample = ["fail safe"]
    g4f_batch_rewrite(sample, model="/nonexistent/model.gguf")
    g4f_batch_translate(sample, model="/nonexistent/model.gguf", target_lang="en", source_lang="ru")
    logger.removeHandler(handler)
    logger.setLevel(previous_level)
    assert captured, "Expected warning logs for missing resources"
    print("OK test_fail_safe_logging")


def test_timestamp_formatting():
    from app.worker import TranscriptionWorker

    assert TranscriptionWorker._format_timestamp(0.0) == "00:00:00.000"
    assert TranscriptionWorker._format_timestamp(3.5) == "00:00:03.500"
    assert TranscriptionWorker._format_timestamp(3723.042) == "01:02:03.042"
    print("OK test_timestamp_formatting")


def test_lmstudio_request_model_load_uses_preset():
    settings = LmStudioSettings(base_url="http://localhost:12345", model="stub")
    client = LmStudioClient(settings)

    calls = []

    class DummyResponse:
        def __init__(self, status: int = 200):
            self.status_code = status

    def fake_post(url, json, timeout):
        calls.append((url, json))
        return DummyResponse()

    client.session.post = fake_post  # type: ignore[attr-defined]

    state = {"checks": 0}

    def fake_find():
        state["checks"] += 1
        if state["checks"] <= 1:
            return {"status": "loading"}
        return {"status": "ready"}

    client._find_model_entry = fake_find  # type: ignore[assignment]

    original_sleep = lmstudio_module.time.sleep
    try:
        lmstudio_module.time.sleep = lambda _: None
        client.ensure_model_loaded(timeout=0.1, poll_interval=0.01)
    finally:
        lmstudio_module.time.sleep = original_sleep

    assert calls, "Model load request not issued"
    preset_values = [payload.get("presetId") for _, payload in calls]
    assert all(value == "1" for value in preset_values), f"Expected preset '1', got {preset_values}"
    print("OK test_lmstudio_request_model_load_uses_preset")


if __name__ == "__main__":
    test_transcription_roundtrip()
    test_rewrite_preserves_line_count()
    test_translate_preserves_line_count()
    test_fail_safe_logging()
    test_timestamp_formatting()
    test_lmstudio_request_model_load_uses_preset()
