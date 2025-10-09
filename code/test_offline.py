"""
Utility checks for the offline pipeline.
Each test prints a short status and exits gracefully when prerequisites are missing.
Run as: python test_offline.py
"""
from __future__ import annotations

import logging
import math
import os
import tempfile
import wave
from datetime import timedelta
from pathlib import Path
from typing import List

import srt

from app.local_llm import g4f_batch_rewrite, g4f_batch_translate


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
    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = Path(tmp_dir) / "sample.wav"
        _generate_sine_wave(audio_path, seconds=10)
        result = model.transcribe(str(audio_path), language="en", verbose=False)
        segments: List[dict] = result.get("segments", [])
        assert segments, "Expected at least one segment"
        txt_path = Path(tmp_dir) / "out.txt"
        srt_path = Path(tmp_dir) / "out.srt"
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
    sample = ["Privet", "kak", "dela", "segodnya", "oshibka tyet"]
    corrected = g4f_batch_rewrite(sample, model="")
    assert len(corrected) == len(sample), "Rewrite changed line count"
    print("OK test_rewrite_preserves_line_count")


def test_translate_preserves_line_count():
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


if __name__ == "__main__":
    test_transcription_roundtrip()
    test_rewrite_preserves_line_count()
    test_translate_preserves_line_count()
    test_fail_safe_logging()
    test_timestamp_formatting()
