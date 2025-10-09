import gc
import os
import threading
from datetime import timedelta
from pathlib import Path
from queue import Empty, Queue
from typing import List, Optional, Tuple

from PyQt6.QtCore import QThread, pyqtSignal
import torch
import whisper
try:
    from moviepy.audio.io.ffmpeg_tools import ffmpeg_extract_audio  # type: ignore
except ModuleNotFoundError:  # moviepy <=1.0.3 compatibility
    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio  # type: ignore
from moviepy.editor import AudioFileClip, VideoFileClip
import srt

from .models import (
    DeviceType,
    OutputRequest,
    ProcessingSettings,
    TranscriptionTask,
)
from .lmstudio_client import (
    LmStudioClient,
    LmStudioError,
    LmStudioSettings,
    chunked,
)

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".oga",
    ".opus",
    ".wma",
}


class TranscriptionWorker(QThread):
    progress_updated = pyqtSignal(str, int)
    task_completed = pyqtSignal(str, object)
    task_failed = pyqtSignal(str, str)
    log_message = pyqtSignal(str, str)
    model_loaded = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.tasks_queue: "Queue[TranscriptionTask]" = Queue()
        self.current_model = None
        self.current_model_size: Optional[str] = None
        self.current_device: Optional[str] = None
        self.current_backend: Optional[str] = None
        self.current_compute_type: Optional[str] = None
        self.settings = ProcessingSettings()
        self._is_running = True
        self._is_processing_paused = False
        self._lock = threading.Lock()
        self._pending_transcripts: List[Tuple[TranscriptionTask, List[dict]]] = []

    def update_processing_settings(self, settings: ProcessingSettings):
        with self._lock:
            self.settings = settings
            self._pending_transcripts.clear()

    def add_task(self, task: TranscriptionTask):
        self.tasks_queue.put(task)

    def stop_processing(self):
        self._is_processing_paused = True
        self.clear_queue()
        with self._lock:
            self._pending_transcripts.clear()
        self._release_model()

    def resume_processing(self):
        self._is_processing_paused = False

    def clear_queue(self):
        with self.tasks_queue.mutex:
            self.tasks_queue.queue.clear()

    def _release_model(self):
        if self.current_model is not None:
            try:
                release = getattr(self.current_model, "__del__", None)
                if callable(release):
                    release()
            except Exception:
                pass
        self.current_model = None
        self.current_model_size = None
        self.current_device = None
        self.current_backend = None
        self.current_compute_type = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _load_openai_whisper(self, size: str, device: str) -> bool:
        resolved_device = device if device in {"cpu", "cuda"} else "cpu"
        if resolved_device == "cuda" and not torch.cuda.is_available():
            resolved_device = "cpu"
        if resolved_device == "cuda" and self.settings.enable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("medium")
        self.log_message.emit(
            "info",
            f"Загрузка Whisper '{size}' на '{resolved_device}'...",
        )
        try:
            self.current_model = whisper.load_model(size, device=resolved_device)
            self.current_model_size = size
            self.current_device = resolved_device
            self.current_backend = "openai"
            self.current_compute_type = None
            self.model_loaded.emit(True)
            self.log_message.emit("success", "Модель загружена.")
            return True
        except RuntimeError as exc:
            if "CUDA" in str(exc) and resolved_device == "cuda":
                self.log_message.emit("warning", f"Ошибка CUDA: {exc}")
                self.log_message.emit("info", "Переключаю на CPU...")
                try:
                    self.current_model = whisper.load_model(size, device="cpu")
                    self.current_model_size = size
                    self.current_device = "cpu"
                    self.current_backend = "openai"
                    self.current_compute_type = None
                    self.model_loaded.emit(True)
                    self.log_message.emit("success", "Модель загружена на CPU.")
                    return True
                except Exception as cpu_error:
                    self.log_message.emit(
                        "error",
                        f"Ошибка загрузки модели на CPU: {cpu_error}",
                    )
            else:
                self.log_message.emit("error", f"Ошибка загрузки модели: {exc}")
        except Exception as exc:
            self.log_message.emit("error", f"Ошибка загрузки модели: {exc}")
        self.model_loaded.emit(False)
        return False

    def _load_faster_whisper(self, size: str, device: str, compute_type: str) -> bool:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            self.log_message.emit(
                "warning",
                "faster-whisper не установлен. Переключаюсь на openai-whisper.",
            )
            return False
        resolved_device = device if device in {"cpu", "cuda"} else "cpu"
        if resolved_device == "cuda" and not torch.cuda.is_available():
            resolved_device = "cpu"
        self.log_message.emit(
            "info",
            f"Загрузка Faster-Whisper '{size}' на '{resolved_device}' ({compute_type})...",
        )
        try:
            self.current_model = WhisperModel(size, device=resolved_device, compute_type=compute_type)
            self.current_model_size = size
            self.current_device = resolved_device
            self.current_backend = "faster"
            self.current_compute_type = compute_type
            self.model_loaded.emit(True)
            self.log_message.emit("success", "Faster-Whisper загружен.")
            return True
        except RuntimeError as exc:
            if "CUDA" in str(exc) and resolved_device == "cuda":
                self.log_message.emit("warning", f"Ошибка Faster-Whisper: {exc}")
                self.log_message.emit("info", "Faster-Whisper на CPU...")
                try:
                    self.current_model = WhisperModel(size, device="cpu", compute_type=compute_type)
                    self.current_model_size = size
                    self.current_device = "cpu"
                    self.current_backend = "faster"
                    self.current_compute_type = compute_type
                    self.model_loaded.emit(True)
                    self.log_message.emit("success", "Faster-Whisper загружен на CPU.")
                    return True
                except Exception as cpu_error:
                    self.log_message.emit(
                        "error",
                        f"Ошибка Faster-Whisper на CPU: {cpu_error}",
                    )
            else:
                self.log_message.emit("error", f"Ошибка Faster-Whisper: {exc}")
        except Exception as exc:
            self.log_message.emit("error", f"Ошибка Faster-Whisper: {exc}")
        self.model_loaded.emit(False)
        return False

    def _load_model(self, backend: str, size: str, device: str, compute_type: str):
        backend = backend or "openai"
        target_device = "cuda" if device == DeviceType.CUDA.value and torch.cuda.is_available() else "cpu"
        if (
            self.current_model
            and self.current_model_size == size
            and self.current_device == target_device
            and self.current_backend == backend
            and (backend != "faster" or self.current_compute_type == (compute_type or "int8"))
        ):
            return
        self._release_model()
        loaded = False
        if backend == "faster":
            loaded = self._load_faster_whisper(size, target_device, compute_type or "int8")
            if not loaded:
                self.log_message.emit("info", "Переключаюсь на openai-whisper.")
        if not loaded:
            loaded = self._load_openai_whisper(size, target_device)
        if not loaded:
            raise RuntimeError("Модель не загружена.")

    def _extract_audio(self, source_path: Path) -> Tuple[Path, bool]:
        ext = source_path.suffix.lower()
        if ext == ".wav":
            return source_path, False
        temp_audio = source_path.with_name(f"{source_path.stem}_temp_audio.wav")
        if ext in AUDIO_EXTENSIONS:
            try:
                ffmpeg_extract_audio(str(source_path), str(temp_audio))
                return temp_audio, True
            except Exception:
                pass
            self.log_message.emit(
                "warning",
                f"FFmpeg не смог сконвертировать {source_path.name}. Пробую внутр. конвертер.",
            )
            with AudioFileClip(str(source_path)) as audio:
                audio.write_audiofile(str(temp_audio), codec="pcm_s16le", logger=None)
            return temp_audio, True
        try:
            ffmpeg_extract_audio(str(source_path), str(temp_audio))
            return temp_audio, True
        except Exception:
            self.log_message.emit(
                "warning",
                f"FFmpeg не смог извлечь аудио из {source_path.name}. Использую moviepy.",
            )
            video = VideoFileClip(str(source_path))
            try:
                video.audio.write_audiofile(str(temp_audio), codec="pcm_s16le", logger=None)
            finally:
                video.close()
            return temp_audio, True

    def _save_as_srt(self, segments: List[dict], output_path: Path):
        srt_segments = [
            srt.Subtitle(
                index=i,
                start=timedelta(seconds=seg["start"]),
                end=timedelta(seconds=seg["end"]),
                content=seg["text"].strip(),
            )
            for i, seg in enumerate(segments, 1)
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt.compose(srt_segments))

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        total_milliseconds = int(round(float(seconds) * 1000))
        hours, remainder = divmod(total_milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, millis = divmod(remainder, 1000)
        return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"

    def _save_as_txt(self, segments: List[dict], output_path: Path, include_timestamps: bool = False):
        with open(output_path, "w", encoding="utf-8") as f:
            for segment in segments:
                text = segment.get("text", "").strip()
                if include_timestamps:
                    start = self._format_timestamp(segment.get("start", 0.0))
                    end = self._format_timestamp(segment.get("end", segment.get("start", 0.0)))
                    f.write(f"{start} --> {end} | {text}\n")
                else:
                    f.write(text + "\n")

    def _save_as_vtt(self, segments: List[dict], output_path: Path):
        def format_vtt(ts: float) -> str:
            total_milliseconds = int(round(float(ts) * 1000))
            hours, remainder = divmod(total_milliseconds, 3_600_000)
            minutes, remainder = divmod(remainder, 60_000)
            secs, millis = divmod(remainder, 1000)
            return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for seg in segments:
                start = format_vtt(seg.get("start", 0.0))
                end = format_vtt(seg.get("end", seg.get("start", 0.0)))
                text = seg.get("text", "").strip()
                f.write(f"{start} --> {end}\n{text}\n\n")

    def _transcribe(self, audio_path: Path, language: str) -> List[dict]:
        if self.current_backend == "faster":
            lang = None if language == "auto" else language
            segments_iter, _ = self.current_model.transcribe(
                str(audio_path),
                language=lang,
                beam_size=5,
                vad_filter=True,
            )
            segments: List[dict] = []
            for seg in segments_iter:
                segments.append(
                    {
                        "start": float(seg.start),
                        "end": float(seg.end),
                        "text": seg.text.strip(),
                    }
                )
            return segments
        result = self.current_model.transcribe(
            str(audio_path),
            language=language if language != "auto" else None,
            fp16=torch.cuda.is_available() and self.current_device == "cuda",
            verbose=False,
        )
        return result.get("segments", []) or []

    def _dispatch_correction(self, task: TranscriptionTask, segments: List[dict]):
        if not task.use_local_llm_correction or not segments:
            self._finalize_task(task, segments)
            return
        if not task.use_lmstudio:
            self.log_message.emit(
                "warning",
                "LM Studio отключён. Коррекция пропущена.",
            )
            self._finalize_task(task, segments)
            return
        with self._lock:
            self._pending_transcripts.append((task, segments))
        self.progress_updated.emit(task.task_id, 85)

    def _process_pending_corrections(self):
        with self._lock:
            pending = list(self._pending_transcripts)
            self._pending_transcripts.clear()
        if not pending:
            return
        client_cache_key = None
        client = None
        for task, segments in pending:
            if not task.use_lmstudio:
                self._finalize_task(task, segments)
                continue
            base_url = (task.lmstudio_base_url or self.settings.lmstudio_base_url or "").strip()
            model = (task.lmstudio_model or self.settings.lmstudio_model or "").strip()
            api_key = (task.lmstudio_api_key or self.settings.lmstudio_api_key or "").strip()
            if not base_url or not model:
                self.log_message.emit("error", "LM Studio не настроен: заполните URL и идентификатор модели.")
                self._finalize_task(task, segments)
                continue
            signature = (base_url, model, api_key)
            if signature != client_cache_key:
                try:
                    client = self._build_lmstudio_client(base_url, model, api_key)
                except ValueError as exc:
                    self.log_message.emit("error", str(exc))
                    self._finalize_task(task, segments)
                    client = None
                    client_cache_key = None
                    continue
                client_cache_key = signature
            if client is None:
                self._finalize_task(task, segments)
                continue
            try:
                self.progress_updated.emit(task.task_id, 90)
                corrected = self._rewrite_with_lmstudio(client, segments, task)
            except LmStudioError as exc:
                self.log_message.emit("error", f"LM Studio ошибка: {exc}")
                corrected = segments
            self._finalize_task(task, corrected)

    def _write_outputs(self, task: TranscriptionTask, segments: List[dict]) -> List[Path]:
        outputs: List[Path] = []
        requested = task.outputs or [
            OutputRequest(format=task.output_format or "srt", include_timestamps=task.include_timestamps)
        ]
        seen_suffixes = set()
        for request in requested:
            fmt = (request.format or "srt").lower()
            base_path = task.output_dir / task.video_path.stem
            if fmt == "srt":
                target = base_path.with_suffix(".srt")
                suffix_key = target.suffix.lower()
                if suffix_key in seen_suffixes:
                    target = base_path.with_name(f"{base_path.name}_{len(outputs)+1}.srt")
                self._save_as_srt(segments, target)
                outputs.append(target)
                seen_suffixes.add(target.suffix.lower())
            elif fmt in {"txt", "txt_plain"}:
                target = base_path.with_suffix(".txt")
                if target.suffix.lower() in seen_suffixes:
                    target = base_path.with_name(f"{base_path.name}_plain.txt")
                self._save_as_txt(segments, target, include_timestamps=False)
                outputs.append(target)
                seen_suffixes.add(target.suffix.lower())
            elif fmt in {"txt_ts", "txt_timestamps"}:
                target = base_path.with_name(f"{base_path.name}_timestamps.txt")
                if target in outputs:
                    target = base_path.with_name(f"{base_path.name}_timestamps_{len(outputs)+1}.txt")
                self._save_as_txt(segments, target, include_timestamps=True)
                outputs.append(target)
                seen_suffixes.add(target.suffix.lower())
            elif fmt == "vtt":
                target = base_path.with_suffix(".vtt")
                if target.suffix.lower() in seen_suffixes:
                    target = base_path.with_name(f"{base_path.name}_{len(outputs)+1}.vtt")
                self._save_as_vtt(segments, target)
                outputs.append(target)
                seen_suffixes.add(target.suffix.lower())
            else:
                target = base_path.with_suffix(f".{fmt}")
                self._save_as_txt(segments, target, include_timestamps=request.include_timestamps)
                outputs.append(target)
                seen_suffixes.add(target.suffix.lower())
        return outputs

    def _finalize_task(self, task: TranscriptionTask, segments: List[dict]):
        try:
            outputs = self._write_outputs(task, segments)
            task.result_paths = outputs
            self.progress_updated.emit(task.task_id, 100)
            self.task_completed.emit(task.task_id, [str(path) for path in outputs])
            self.log_message.emit(
                "success",
                f"Transcription completed: {task.video_path.name} ({len(outputs)} file(s)).",
            )
        except Exception as exc:
            self.task_failed.emit(task.task_id, str(exc))
            self.log_message.emit(
                "error",
                f"Failed to save results for {task.video_path.name}: {exc}",
            )

    def _build_lmstudio_client(self, base_url: str, model: str, api_key: str) -> LmStudioClient:
        settings = LmStudioSettings(
            base_url=base_url.rstrip("/"),
            model=model,
            api_key=api_key,
            timeout=120.0,
            temperature=0.1,
        )
        return LmStudioClient(settings)

    def _rewrite_with_lmstudio(
        self,
        client: LmStudioClient,
        segments: List[dict],
        task: TranscriptionTask,
    ) -> List[dict]:
        texts = [seg.get("text", "").strip() for seg in segments]
        if not texts:
            return segments
        batch_size = max(1, task.lmstudio_batch_size or self.settings.lmstudio_batch_size or 40)
        lang_hint = "" if task.language == "auto" else (task.language or "")
        rewritten: list[str] = []
        for chunk in chunked(texts, batch_size):
            rewritten.extend(client.rewrite_batch(chunk, lang_hint))
        if len(rewritten) != len(segments):
            self.log_message.emit("warning", "LM Studio returned unexpected number of lines. Keeping original text.")
            return segments
        for seg, new_text in zip(segments, rewritten):
            seg["text"] = new_text
        return segments

    def _process_task(self, task: TranscriptionTask):
        audio_path: Optional[Path] = None
        cleanup = False
        try:
            self.log_message.emit("info", f"Starting transcription: {task.video_path.name}")
            self.progress_updated.emit(task.task_id, 5)
            self._load_model(
                task.whisper_backend or "openai",
                task.model_size,
                task.device,
                task.faster_whisper_compute_type or "int8",
            )
            self.progress_updated.emit(task.task_id, 15)
            audio_path, cleanup = self._extract_audio(task.video_path)
            self.progress_updated.emit(task.task_id, 30)
            self.log_message.emit("info", f"Processing audio {audio_path.name}...")
            segments = self._transcribe(audio_path, task.language)
            self.progress_updated.emit(task.task_id, 70)
            self._dispatch_correction(task, segments)
        except Exception as exc:
            self.task_failed.emit(task.task_id, str(exc))
            self.log_message.emit("error", f"Transcription error for {task.video_path.name}: {exc}")
        finally:
            if cleanup and audio_path and audio_path.exists():
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass
    def run(self):
        while self._is_running:
            if self._is_processing_paused:
                self.msleep(150)
                continue
            try:
                task = self.tasks_queue.get(timeout=0.05)
            except Empty:
                self._process_pending_corrections()
                self.msleep(50)
                continue
            try:
                self._process_task(task)
            finally:
                self.tasks_queue.task_done()
            if self.tasks_queue.empty():
                self._release_model()
                self._process_pending_corrections()
            self.msleep(15)
        self._process_pending_corrections()

    def stop(self):
        self._is_running = False
        self.clear_queue()
        with self._lock:
            self._pending_transcripts.clear()
        self._release_model()

