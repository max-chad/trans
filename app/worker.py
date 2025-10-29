import gc
import ctypes
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, List, Optional, Tuple

_CUDA_PATHS_CONFIGURED = False
_CUDA_DLL_HANDLES: List[object] = []
_CUDA_DLL_PATHS: List[str] = []
_CUDA_PATH_LOGGED = False

# Добавляем CUDA-библиотеки в системные пути, чтобы Whisper мог работать на GPU в Windows.


def _configure_cuda_dll_search_path() -> None:
    global _CUDA_PATHS_CONFIGURED
    if _CUDA_PATHS_CONFIGURED or os.name != "nt":
        return
    base = Path(sys.prefix) if hasattr(sys, "prefix") else None
    candidates: list[Path] = []
    cuda_root: Optional[Path] = None
    if base:
        site_root = base / "Lib" / "site-packages" / "nvidia"
        cuda_root = site_root
        for subdir in ("cublas", "cudnn", "cuda_runtime"):
            candidates.append(site_root / subdir / "bin")
            candidates.append(site_root / subdir / "lib" / "x64")
        torch_lib = base / "Lib" / "site-packages" / "torch" / "lib"
        candidates.append(torch_lib)
        # Some wheel builds place CUDA DLLs directly under torch root.
        candidates.append(torch_lib.parent)
    env_path = os.environ.get("PATH", "")
    updated_env = env_path
    seen = {part for part in env_path.split(os.pathsep) if part}
    for directory in candidates:
        if not directory.is_dir():
            continue
        directory_str = str(directory)
        try:
            handle = os.add_dll_directory(directory_str)
            _CUDA_DLL_HANDLES.append(handle)
            _CUDA_DLL_PATHS.append(directory_str)
        except AttributeError:
            if directory_str not in seen:
                updated_env = directory_str + os.pathsep + updated_env
                seen.add(directory_str)
                _CUDA_DLL_PATHS.append(directory_str)
    if updated_env != env_path:
        os.environ["PATH"] = updated_env
        # Propagate CUDA_PATH for libraries that inspect it explicitly.
        target_root = cuda_root or base
        if target_root:
            os.environ.setdefault("CUDA_PATH", str(target_root))
    for lib_name in (
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cudart64_12.dll",
        "cusolver64_11.dll",
        "cusparse64_12.dll",
    ):
        try:
            ctypes.WinDLL(lib_name)
        except OSError:
            # Leave detailed logging to the worker when the issue manifests.
            continue
    _CUDA_PATHS_CONFIGURED = True


if os.name == "nt":
    _configure_cuda_dll_search_path()

import torch
import whisper
from PyQt6.QtCore import QThread, pyqtSignal
try:
    from moviepy.audio.io.ffmpeg_tools import ffmpeg_extract_audio  # type: ignore
except ModuleNotFoundError:  # moviepy <=1.0.3 compatibility
    try:
        from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio  # type: ignore
    except ModuleNotFoundError:
        ffmpeg_extract_audio = None  # type: ignore
try:
    from moviepy.editor import AudioFileClip, VideoFileClip  # type: ignore
except ModuleNotFoundError:
    AudioFileClip = None  # type: ignore
    VideoFileClip = None  # type: ignore
import srt

from .models import (
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

# Контекст вычислительного устройства с очередью задач и загруженной моделью.
@dataclass
class _DeviceContext:
    device: str
    queue: "Queue[Optional[TranscriptionTask]]" = field(default_factory=Queue)
    thread: Optional[threading.Thread] = None
    model: Optional[object] = None
    model_size: Optional[str] = None
    backend: Optional[str] = None
    compute_type: Optional[str] = None
    runtime_device: str = "cpu"
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    active: bool = False


# Допустимые расширения файлов, из которых можно извлекать аудио.
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


# Запускаем ffmpeg через CLI и конвертируем входной файл в WAV.
def _run_ffmpeg_cli(input_path: Path, output_path: Path) -> bool:
    executable = shutil.which("ffmpeg")
    if not executable:
        return False
    if output_path.exists():
        try:
            output_path.unlink()
        except OSError:
            return False
    cmd = [
        executable,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return result.returncode == 0 and output_path.exists()


# Пробуем извлечь аудио через moviepy, затем через ffmpeg как резерв.
def _extract_with_ffmpeg(input_path: Path, output_path: Path) -> bool:
    if output_path.exists():
        try:
            output_path.unlink()
        except OSError:
            return False
    if ffmpeg_extract_audio is not None:
        try:
            ffmpeg_extract_audio(str(input_path), str(output_path))
            return output_path.exists()
        except Exception:
            pass
    return _run_ffmpeg_cli(input_path, output_path)


# Рабочий поток, который принимает задачи транскрибации и управляет выводом результатов.
class TranscriptionWorker(QThread):
    progress_updated = pyqtSignal(str, int)
    task_completed = pyqtSignal(str, object)
    task_failed = pyqtSignal(str, str)
    log_message = pyqtSignal(str, str)
    model_loaded = pyqtSignal(bool)

    # Инициализируем рабочий поток и все очереди.
    def __init__(self):
        super().__init__()
        _configure_cuda_dll_search_path()
        # Настраиваем очереди, контексты устройств и вспомогательные потоки.
        self.tasks_queue: "Queue[TranscriptionTask]" = Queue()
        self.settings = ProcessingSettings()
        self._is_running = True
        self._is_processing_paused = False
        self._lock = threading.Lock()
        self._ffmpeg_checked = False
        self._ffmpeg_available = False
        self._contexts: Dict[str, _DeviceContext] = {"cpu": _DeviceContext("cpu")}
        if torch.cuda.is_available():
            self._contexts["cuda"] = _DeviceContext("cuda")
        self._hybrid_last_device = "cuda"
        self._shutdown_event = threading.Event()
        self._correction_shutdown = threading.Event()
        self._correction_queue: "Queue[Optional[Tuple[TranscriptionTask, List[dict]]]]" = Queue()
        self._correction_thread = threading.Thread(target=self._correction_worker, daemon=True)
        self._correction_thread.start()
        self._correction_waiters: Dict[str, threading.Event] = {}
        self._task_start_times: Dict[str, float] = {}
        self._processing_start_time: Optional[float] = None
        self._cumulative_task_time: float = 0.0
        self._completed_task_count: int = 0
        self._staged_mode_enabled = False
        self._staged_corrections: List[TranscriptionTask] = []
        self._in_correction_phase = False
        self._active_transcriptions = 0

    # Обновляем параметры обработки и при необходимости запускаем или завершаем фазу корректировки.
    def update_processing_settings(self, settings: ProcessingSettings):
        pending_flush: List[TranscriptionTask] = []
        with self._lock:
            previous_staged = self._staged_mode_enabled
            self.settings = settings
            self._staged_mode_enabled = (settings.pipeline_mode or "").strip().lower() == "staged"
            if previous_staged and not self._staged_mode_enabled:
                pending_flush = list(self._staged_corrections)
                self._staged_corrections.clear()
                self._in_correction_phase = False
        if pending_flush:
            for task in pending_flush:
                self._dispatch_correction(task)
        if self._staged_mode_enabled:
            self._maybe_start_correction_phase()

    # Помещаем новую задачу транскрибации в очередь основного потока.
    def add_task(self, task: TranscriptionTask):
        self.tasks_queue.put(task)

    # Временно приостанавливаем обработку и освобождаем ресурсы.
    def stop_processing(self):
        self._is_processing_paused = True
        self.clear_queue()
        self._drain_correction_queue()
        self._release_all_models()

    # Снимаем паузу с обработки.
    def resume_processing(self):
        self._is_processing_paused = False

    # Полностью очищаем очереди задач и сбрасываем состояние staged-режима.
    def clear_queue(self):
        with self.tasks_queue.mutex:
            self.tasks_queue.queue.clear()
        for context in self._contexts.values():
            with context.queue.mutex:
                context.queue.queue.clear()
        self._drain_correction_queue()
        pending: List[TranscriptionTask] = []
        with self._lock:
            pending = list(self._staged_corrections)
            self._staged_corrections.clear()
            self._in_correction_phase = False
            self._active_transcriptions = 0
        for task in pending:
            self._cleanup_staged_transcript(task)

    # Сбрасываем и выгружаем модель из контекста, очищая память устройства.
    def _release_context_model(self, context: _DeviceContext):
        with context.lock:
            if context.model is not None:
                try:
                    release = getattr(context.model, "__del__", None)
                    if callable(release):
                        release()
                except Exception:
                    pass
            context.model = None
            context.model_size = None
            context.backend = None
            context.compute_type = None
            if context.runtime_device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            context.runtime_device = context.device
        gc.collect()

    # Освобождаем модели во всех контекстах устройств.
    def _release_all_models(self):
        for context in self._contexts.values():
            self._release_context_model(context)

    # Загружаем пайплайн diarization от pyannote и кэшируем его для повторного использования.
    @staticmethod
    # Объединяем соседние сегменты с одинаковым говорящим в один блок.
    def _group_segments_by_speaker(segments: List[dict]) -> List[dict]:
        grouped: List[dict] = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            speaker = seg.get("speaker")
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", start) or start)
            if grouped and grouped[-1].get("speaker") == speaker:
                grouped[-1]["text"] = (grouped[-1]["text"] + " " + text).strip()
                grouped[-1]["end"] = end
            else:
                grouped.append(
                    {
                        "speaker": speaker,
                        "start": start,
                        "end": end,
                        "text": text,
                    }
                )
        return grouped

    # Запускаем diarization, обогащая сегменты Whisper информацией о говорящих.
    def _apply_speaker_diarization(
        self,
        task: TranscriptionTask,
        audio_path: Optional[Path],
        segments: List[dict],
    ) -> List[dict]:
        return segments

    # Фиксируем старт обработки задачи для дальнейшей статистики.
    def _mark_task_start(self, task: TranscriptionTask):
        now = time.perf_counter()
        with self._lock:
            if self._processing_start_time is None:
                self._processing_start_time = now
            self._task_start_times.setdefault(task.task_id, now)

    # Логируем длительность выполнения задачи и агрегированную статистику.
    def _log_task_timing(self, task: TranscriptionTask, status: str):
        now = time.perf_counter()
        with self._lock:
            start = self._task_start_times.pop(task.task_id, None)
            if start is None:
                start = now
            duration = max(0.0, now - start)
            self._cumulative_task_time += duration
            self._completed_task_count += 1
            cumulative = self._cumulative_task_time
            completed = self._completed_task_count
        duration_text = self._format_duration(duration)
        total_text = self._format_duration(cumulative)
        status_text = "completed" if status == "completed" else "failed"
        self.log_message.emit(
            "info",
            f"Timing: {task.video_path.name} {status_text} in {duration_text}. "
            f"Total task time: {total_text} across {completed} task(s).",
        )

    @staticmethod
    # Преобразуем длительность в человекочитаемый формат HH:MM:SS.mmm.
    def _format_duration(seconds: float) -> str:
        if seconds <= 0:
            return "00:00:00.000"
        total_milliseconds = int(round(seconds * 1000))
        hours, remainder = divmod(total_milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, millis = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    # Сообщаем ожидающим потокам, что корректировка конкретной задачи завершена.
    def _notify_correction_done(self, task_id: str):
        waiter: Optional[threading.Event]
        with self._lock:
            waiter = self._correction_waiters.pop(task_id, None)
        if waiter:
            waiter.set()
        if self._staged_mode_enabled:
            self._check_correction_phase_completion()

    # Останавливаем поток корректировок и освобождаем его очередь.
    def _shutdown_correction_worker(self):
        if not self._correction_shutdown.is_set():
            self._correction_shutdown.set()
            self._correction_queue.put(None)

    @staticmethod
    # Проверяем, сигнализирует ли сообщение об ошибке о проблемах с CUDA-библиотеками.
    def _is_cuda_dependency_error(message: str) -> bool:
        lowered = message.lower()
        keywords = [
            "cuda",
            "cublas",
            "cudnn",
            "libcudart",
            "torch not compiled with cuda",
            "libcusparse",
            "libcublas",
        ]
        return any(keyword in lowered for keyword in keywords)

    @staticmethod
    # Определяем типичную ошибку Faster-Whisper о несоответствии формы признаков.
    def _is_feature_shape_error(message: str) -> bool:
        lowered = message.lower()
        return "invalid input features shape" in lowered and "expected an input with shape" in lowered

    # Выгребаем очереди корректировок, уведомляя ожидающих при отмене.
    def _drain_correction_queue(self):
        while True:
            try:
                item = self._correction_queue.get_nowait()
            except Empty:
                break
            else:
                if item and isinstance(item, tuple):
                    task = item[0]
                    if isinstance(task, TranscriptionTask):
                        self._notify_correction_done(task.task_id)
                        self._cleanup_staged_transcript(task)
                self._correction_queue.task_done()

    # Увеличиваем счётчик одновременно выполняемых транскрибаций.
    def _increment_active_transcriptions(self):
        with self._lock:
            self._active_transcriptions += 1

    # Уменьшаем счётчик активных транскрибаций и запускаем коррекцию, если все завершились.
    def _decrement_active_transcriptions(self):
        with self._lock:
            if self._active_transcriptions > 0:
                self._active_transcriptions -= 1
            remaining = self._active_transcriptions
        if remaining == 0:
            self._maybe_start_correction_phase()

    # Проверяем, остались ли активные или ожидающие задачи на любом устройстве.
    def _context_has_pending_transcriptions(self) -> bool:
        for context in self._contexts.values():
            if context.active or not context.queue.empty():
                return True
        return False

    # Если включен staged-режим, решаем, пора ли переходить к фазе корректировки.
    def _maybe_start_correction_phase(self):
        if not self._staged_mode_enabled:
            return
        pending: List[TranscriptionTask] = []
        with self._lock:
            if self._in_correction_phase:
                return
            if self._active_transcriptions > 0:
                return
            if not self._staged_corrections:
                return
            if not self._correction_queue.empty():
                return
            if not self.tasks_queue.empty() or self._context_has_pending_transcriptions():
                return
            pending = list(self._staged_corrections)
            self._staged_corrections.clear()
            self._in_correction_phase = True
        if pending:
            self.log_message.emit(
                "info",
                f"Starting staged correction phase for {len(pending)} task(s).",
            )
        for task in pending:
            self._dispatch_correction(task)

    # Проверяем, можно ли завершить текущую фазу корректировки и перейти к следующей.
    def _check_correction_phase_completion(self):
        if not self._staged_mode_enabled:
            return
        with self._lock:
            if not self._in_correction_phase:
                return
            if self._staged_corrections:
                return
            if not self._correction_queue.empty():
                return
            self._in_correction_phase = False
        self._maybe_start_correction_phase()

    # Решаем, отправлять ли сегменты на локальную корректировку LLM или завершать задачу сразу.
    def _handle_post_transcription(self, task: TranscriptionTask, segments: List[dict]):
        if not segments or not task.use_local_llm_correction:
            self._finalize_task(task, segments)
            return
        if not self._staged_mode_enabled or not task.use_lmstudio:
            self._dispatch_correction(task, segments)
            return
        self.progress_updated.emit(task.task_id, 75)
        staged_path = self._create_staged_transcript(task, segments)
        if staged_path is None:
            self._dispatch_correction(task, segments)
            return
        with self._lock:
            self._staged_corrections.append(task)
        self._maybe_start_correction_phase()

    def _create_staged_transcript(self, task: TranscriptionTask, segments: List[dict]) -> Optional[Path]:
        self._cleanup_staged_transcript(task)
        staging_dir = Path(tempfile.gettempdir()) / "video-transcriber-staged"
        try:
            staging_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.log_message.emit(
                "warning",
                f"Failed to prepare staging directory for {task.video_path.name}: {exc}",
            )
            return None
        target = staging_dir / f"{task.task_id}.json"
        try:
            with target.open("w", encoding="utf-8") as handle:
                json.dump(segments, handle, ensure_ascii=False)
        except Exception as exc:
            self.log_message.emit(
                "warning",
                f"Failed to write staged transcript for {task.video_path.name}: {exc}",
            )
            if target.exists():
                try:
                    target.unlink()
                except OSError:
                    pass
            return None
        task.staged_transcript_path = target
        return target

    def _load_staged_segments(self, task: TranscriptionTask) -> List[dict]:
        path = getattr(task, "staged_transcript_path", None)
        if not path:
            return []
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            self.log_message.emit(
                "warning",
                f"Staged transcript missing for {task.video_path.name}. Continuing without correction.",
            )
            return []
        except json.JSONDecodeError as exc:
            self.log_message.emit(
                "warning",
                f"Failed to parse staged transcript for {task.video_path.name}: {exc}",
            )
            return []
        if not isinstance(data, list):
            self.log_message.emit(
                "warning",
                f"Unexpected staged transcript format for {task.video_path.name}.",
            )
            return []
        segments: List[dict] = []
        for item in data:
            if isinstance(item, dict):
                segments.append(item)
        return segments

    def _cleanup_staged_transcript(self, task: TranscriptionTask):
        path = getattr(task, "staged_transcript_path", None)
        if not path:
            return
        try:
            Path(path).unlink(missing_ok=True)  # type: ignore[arg-type]
        except TypeError:
            # Python <3.8 compatibility for missing_ok.
            try:
                target = Path(path)
                if target.exists():
                    target.unlink()
            except OSError:
                pass
        except OSError:
            pass
        task.staged_transcript_path = None

    # Поток, применяющий локальную LLM-коррекцию к уже транскрибированным сегментам.
    def _correction_worker(self):
        client_cache_key = None
        client: Optional[LmStudioClient] = None
        while not self._correction_shutdown.is_set():
            try:
                item = self._correction_queue.get(timeout=0.1)
            except Empty:
                continue
            if item is None:
                self._correction_queue.task_done()
                break
            task, segments = item
            finalized = False
            try:
                if not task.use_local_llm_correction or not segments:
                    self._finalize_task(task, segments)
                    finalized = True
                    continue
                if not task.use_lmstudio:
                    self.log_message.emit(
                        "warning",
                        "LM Studio отключён. Коррекция пропущена.",
                    )
                    self._finalize_task(task, segments)
                    finalized = True
                    continue
                base_url = (task.lmstudio_base_url or self.settings.lmstudio_base_url or "").strip()
                model = (task.lmstudio_model or self.settings.lmstudio_model or "").strip()
                api_key = (task.lmstudio_api_key or self.settings.lmstudio_api_key or "").strip()
                if not base_url or not model:
                    self.log_message.emit("error", "LM Studio не настроен: заполните URL и идентификатор модели.")
                    self._finalize_task(task, segments)
                    finalized = True
                    continue
                prompt_limit = (
                    task.lmstudio_prompt_token_limit
                    or self.settings.lmstudio_prompt_token_limit
                    or 0
                )
                response_limit = (
                    task.lmstudio_response_token_limit
                    or self.settings.lmstudio_response_token_limit
                    or 0
                )
                token_margin = (
                    task.lmstudio_token_margin
                    or self.settings.lmstudio_token_margin
                    or 0
                )
                load_timeout = (
                    getattr(task, "lmstudio_load_timeout", None)
                    or self.settings.lmstudio_load_timeout
                    or 0
                )
                poll_interval = (
                    getattr(task, "lmstudio_poll_interval", None)
                    or self.settings.lmstudio_poll_interval
                    or 0
                )
                signature = (base_url, model, api_key, prompt_limit, response_limit, token_margin, load_timeout, poll_interval)
                if signature != client_cache_key:
                    try:
                        candidate = self._build_lmstudio_client(
                            base_url,
                            model,
                            api_key,
                            prompt_limit,
                            response_limit,
                            token_margin,
                        )
                        candidate.ensure_model_loaded(
                            lambda level, message: self.log_message.emit(level, message),
                            timeout=float(load_timeout) if load_timeout else 300.0,
                            poll_interval=float(poll_interval) if poll_interval else 1.5,
                        )
                    except (ValueError, LmStudioError) as exc:
                        self.log_message.emit("error", str(exc))
                        self._finalize_task(task, segments)
                        finalized = True
                        client = None
                        client_cache_key = None
                        continue
                    client = candidate
                    client_cache_key = signature
                if client is None:
                    self._finalize_task(task, segments)
                    finalized = True
                    continue
                try:
                    self.progress_updated.emit(task.task_id, 90)
                    corrected = self._rewrite_with_lmstudio(client, segments, task)
                except LmStudioError as exc:
                    self.log_message.emit("error", f"LM Studio ошибка: {exc}")
                    corrected = segments
                self._finalize_task(task, corrected)
                finalized = True
            except Exception as exc:
                self.log_message.emit(
                    "error",
                    f"Неожиданная ошибка коррекции ({exc.__class__.__name__}): {exc}",
                )
                stack = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                if stack:
                    self.log_message.emit("info", stack.strip())
                if not finalized:
                    self._finalize_task(task, segments)
                    finalized = True
            finally:
                self._correction_queue.task_done()

    # Загружаем модель openai-whisper в заданный контекст, учитывая доступность CUDA.
    def _load_openai_whisper(self, context: _DeviceContext, size: str) -> bool:
        target_device = context.device if context.device in {"cpu", "cuda"} else "cpu"
        if target_device == "cuda" and not torch.cuda.is_available():
            target_device = "cpu"
        if target_device == "cuda" and self.settings.enable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("medium")
        self.log_message.emit(
            "info",
            f"Loading Whisper '{size}' on '{target_device}'...",
        )
        try:
            model = whisper.load_model(size, device=target_device)
            with context.lock:
                context.model = model
                context.model_size = size
                context.backend = "openai"
                context.compute_type = None
                context.runtime_device = target_device
            self.model_loaded.emit(True)
            self.log_message.emit("success", "Model loaded.")
            return True
        except RuntimeError as exc:
            if "CUDA" in str(exc) and target_device == "cuda":
                self.log_message.emit("warning", f"CUDA error: {exc}")
                self.log_message.emit("info", "Retrying on CPU...")
                try:
                    model = whisper.load_model(size, device="cpu")
                    with context.lock:
                        context.model = model
                        context.model_size = size
                        context.backend = "openai"
                        context.compute_type = None
                        context.runtime_device = "cpu"
                    self.model_loaded.emit(True)
                    self.log_message.emit("success", "Model loaded on CPU.")
                    return True
                except Exception as cpu_error:
                    self.log_message.emit(
                        "error",
                        f"Failed to load model on CPU: {cpu_error}",
                    )
            else:
                self.log_message.emit("error", f"Failed to load model: {exc}")
        except Exception as exc:
            self.log_message.emit("error", f"Failed to load model: {exc}")
        self.model_loaded.emit(False)
        return False

    # Загружаем faster-whisper и обрабатываем падение на CPU при ошибках CUDA.
    def _load_faster_whisper(self, context: _DeviceContext, size: str, compute_type: str) -> bool:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            self.log_message.emit("warning", "faster-whisper is not installed. Falling back to openai-whisper.")
            return False
        target_device = context.device if context.device in {"cpu", "cuda"} else "cpu"
        if target_device == "cuda" and not torch.cuda.is_available():
            target_device = "cpu"
        self.log_message.emit(
            "info",
            f"Loading Faster-Whisper '{size}' on '{target_device}' ({compute_type})...",
        )
        try:
            model = WhisperModel(size, device=target_device, compute_type=compute_type)
            with context.lock:
                context.model = model
                context.model_size = size
                context.backend = "faster"
                context.compute_type = compute_type
                context.runtime_device = target_device
            self.model_loaded.emit(True)
            self.log_message.emit("success", "Faster-Whisper loaded.")
            return True
        except RuntimeError as exc:
            message = str(exc)
            if target_device == "cuda" and self._is_cuda_dependency_error(message):
                self.log_message.emit("warning", f"Faster-Whisper CUDA error: {message}")
                self.log_message.emit("info", "Retrying Faster-Whisper on CPU...")
                try:
                    model = WhisperModel(size, device="cpu", compute_type=compute_type)
                    with context.lock:
                        context.model = model
                        context.model_size = size
                        context.backend = "faster"
                        context.compute_type = compute_type
                        context.runtime_device = "cpu"
                    self.model_loaded.emit(True)
                    self.log_message.emit("success", "Faster-Whisper loaded on CPU.")
                    return True
                except Exception as cpu_error:
                    self.log_message.emit(
                        "error",
                        f"Faster-Whisper failed on CPU: {cpu_error}",
                    )
            else:
                self.log_message.emit("error", f"Faster-Whisper error: {message}")
        except Exception as exc:
            message = str(exc)
            if target_device == "cuda" and self._is_cuda_dependency_error(message):
                self.log_message.emit("warning", f"Faster-Whisper CUDA error: {message}")
                self.log_message.emit("info", "Retrying Faster-Whisper on CPU...")
                try:
                    model = WhisperModel(size, device="cpu", compute_type=compute_type)
                    with context.lock:
                        context.model = model
                        context.model_size = size
                        context.backend = "faster"
                        context.compute_type = compute_type
                        context.runtime_device = "cpu"
                    self.model_loaded.emit(True)
                    self.log_message.emit("success", "Faster-Whisper loaded on CPU.")
                    return True
                except Exception as cpu_error:
                    self.log_message.emit(
                        "error",
                        f"Faster-Whisper failed on CPU: {cpu_error}",
                    )
            else:
                self.log_message.emit("error", f"Faster-Whisper error: {message}")
        self.model_loaded.emit(False)
        return False

    # Гарантируем, что нужная модель загружена в контекст; при несоответствии переинициализируем.
    def _ensure_model_for_context(self, context: _DeviceContext, backend: str, size: str, compute_type: str) -> bool:
        desired_backend = (backend or "openai").lower()
        normalized_compute = (compute_type or "int8").lower()
        current_backend = (context.backend or "").lower()
        if (
            context.model is not None
            and context.model_size == size
            and current_backend == desired_backend
            and (desired_backend != "faster" or (context.compute_type or "").lower() == normalized_compute)
        ):
            return True
        self._release_context_model(context)
        loaded = False
        if desired_backend == "faster":
            loaded = self._load_faster_whisper(context, size, normalized_compute)
            if not loaded:
                self.log_message.emit("info", "Falling back to openai-whisper.")
        if not loaded:
            loaded = self._load_openai_whisper(context, size)
        return loaded

    # Извлекаем WAV-дорожку из видео или аудио-файла, используя ffmpeg и moviepy как резерв.
    def _extract_audio(self, source_path: Path) -> Tuple[Path, bool]:
        ext = source_path.suffix.lower()
        if ext == ".wav":
            return source_path, False
        temp_audio = source_path.with_name(f"{source_path.stem}_temp_audio.wav")

        if _extract_with_ffmpeg(source_path, temp_audio):
            return temp_audio, True

        if ext in AUDIO_EXTENSIONS:
            if AudioFileClip is not None:
                try:
                    with AudioFileClip(str(source_path)) as audio:
                        audio.write_audiofile(str(temp_audio), codec="pcm_s16le", logger=None)
                    return temp_audio, True
                except Exception as exc:
                    self.log_message.emit(
                        "warning",
                        f"MoviePy failed to export audio from {source_path.name}: {exc}",
                    )
            else:
                self.log_message.emit(
                    "warning",
                    "MoviePy is not installed; skipping AudioFileClip fallback.",
                )
            if _run_ffmpeg_cli(source_path, temp_audio):
                return temp_audio, True
        else:
            if VideoFileClip is not None:
                try:
                    video = VideoFileClip(str(source_path))
                    try:
                        audio = video.audio
                        if audio is None:
                            raise ValueError("Video has no audio track.")
                        audio.write_audiofile(str(temp_audio), codec="pcm_s16le", logger=None)
                        return temp_audio, True
                    finally:
                        video.close()
                except Exception as exc:
                    self.log_message.emit(
                        "warning",
                        f"MoviePy failed to extract audio track from {source_path.name}: {exc}",
                    )
            else:
                self.log_message.emit(
                    "warning",
                    "MoviePy is not installed; skipping VideoFileClip fallback.",
                )
            if _run_ffmpeg_cli(source_path, temp_audio):
                return temp_audio, True

        raise RuntimeError(
            f"Failed to extract audio from {source_path.name}. "
            "Install MoviePy or ensure ffmpeg is available in PATH."
        )

    # Сохраняем сегменты в формате .srt с поддержкой имён говорящих.
    def _save_as_srt(self, segments: List[dict], output_path: Path):
        srt_segments: List[srt.Subtitle] = []
        for i, seg in enumerate(segments, 1):
            text = seg.get("text", "").strip()
            if not text:
                continue
            speaker = seg.get("speaker")
            if speaker:
                text = f"{speaker}: {text}"
            srt_segments.append(
                srt.Subtitle(
                    index=i,
                    start=timedelta(seconds=float(seg.get("start", 0.0) or 0.0)),
                    end=timedelta(seconds=float(seg.get("end", seg.get("start", 0.0)) or 0.0)),
                    content=text,
                )
            )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt.compose(srt_segments))

    @staticmethod
    # Форматируем секунды в метку времени, совместимую с .txt/.vtt.
    def _format_timestamp(seconds: float) -> str:
        total_milliseconds = int(round(float(seconds) * 1000))
        hours, remainder = divmod(total_milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, millis = divmod(remainder, 1000)
        return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"

    # Сохраняем транскрипт в текстовом виде, опционально добавляя таймкоды.
    def _save_as_txt(self, segments: List[dict], output_path: Path, include_timestamps: bool = False):
        with open(output_path, "w", encoding="utf-8") as f:
            if include_timestamps:
                for segment in segments:
                    text = segment.get("text", "").strip()
                    if not text:
                        continue
                    speaker = segment.get("speaker")
                    prefix = f"{speaker}: " if speaker else ""
                    start = self._format_timestamp(segment.get("start", 0.0))
                    end = self._format_timestamp(segment.get("end", segment.get("start", 0.0)))
                    f.write(f"{start} --> {end} | {prefix}{text}\n")
            else:
                has_speakers = any(seg.get("speaker") for seg in segments)
                blocks = (
                    self._group_segments_by_speaker(segments) if has_speakers else [
                        {
                            "speaker": seg.get("speaker"),
                            "text": seg.get("text", "").strip(),
                        }
                        for seg in segments
                        if seg.get("text", "").strip()
                    ]
                )
                first_block = True
                for block in blocks:
                    text = block.get("text", "").strip()
                    if not text:
                        continue
                    speaker = block.get("speaker")
                    if speaker:
                        if not first_block:
                            f.write("\n")
                        f.write(f"{speaker}:\n{text}\n")
                    else:
                        f.write(text + "\n")
                    first_block = False
                if first_block:
                    f.write("")

    # Генерируем WebVTT файл из сегментов Whisper.
    def _save_as_vtt(self, segments: List[dict], output_path: Path):
        # Локальный форматтер для меток времени WebVTT.
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
                if not text:
                    continue
                speaker = seg.get("speaker")
                if speaker:
                    text = f"{speaker}: {text}"
                f.write(f"{start} --> {end}\n{text}\n\n")

    # Выполняем транскрибацию аудио выбранным бэкендом и возвращаем список сегментов.
    def _transcribe(self, context: _DeviceContext, audio_path: Path, language: str) -> List[dict]:
        model = context.model
        if model is None:
            return []
        backend = (context.backend or "").lower()
        if backend == "faster":
            lang = None if language == "auto" else language
            segments_iter, _ = model.transcribe(
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
        result = model.transcribe(
            str(audio_path),
            language=language if language != "auto" else None,
            fp16=torch.cuda.is_available() and context.runtime_device == "cuda",
            verbose=False,
        )
        return result.get("segments", []) or []

    # Отправляем сегменты в очередь локальной LLM-коррекции (или завершаем задачу сразу).
    def _dispatch_correction(self, task: TranscriptionTask, segments: Optional[List[dict]] = None):
        if segments is None:
            segments = self._load_staged_segments(task)
        if not segments:
            segments = []
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
        mode = (task.pipeline_mode or self.settings.pipeline_mode or "balanced").strip().lower()
        wait_for_serial = mode == "serial"
        waiter: Optional[threading.Event] = None
        if wait_for_serial:
            waiter = threading.Event()
            with self._lock:
                self._correction_waiters[task.task_id] = waiter
        self.progress_updated.emit(task.task_id, 85)
        self._correction_queue.put((task, segments))
        if waiter:
            waiter.wait()

    # Сохраняем выходные файлы в запрошенных пользователем форматах.
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

    # Финализируем задачу: пишем файлы, уведомляем UI и логируем статистику.
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
            self._log_task_timing(task, "completed")
        except Exception as exc:
            self.task_failed.emit(task.task_id, str(exc))
            self.log_message.emit(
                "error",
                f"Failed to save results for {task.video_path.name}: {exc}",
            )
            self._log_task_timing(task, "failed")
        finally:
            self._cleanup_staged_transcript(task)
            self._notify_correction_done(task.task_id)

    # Строим клиент для LM Studio с учётом пользовательских ограничений по токенам.
    def _build_lmstudio_client(
        self,
        base_url: str,
        model: str,
        api_key: str,
        prompt_limit: int,
        response_limit: int,
        token_margin: int,
    ) -> LmStudioClient:
        settings = LmStudioSettings(
            base_url=base_url.rstrip("/"),
            model=model,
            api_key=api_key,
            timeout=120.0,
            temperature=0.1,
            max_completion_tokens=response_limit or 4096,
            max_prompt_tokens=prompt_limit or 8192,
            prompt_token_margin=max(0, token_margin),
        )
        return LmStudioClient(settings)

    # Отправляем сегменты в LM Studio для пост-обработки и применяем результат.
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
        prompt_limit = (
            task.lmstudio_prompt_token_limit
            or self.settings.lmstudio_prompt_token_limit
            or 0
        )
        token_margin = (
            task.lmstudio_token_margin
            or self.settings.lmstudio_token_margin
            or 0
        )
        deep_mode = bool(getattr(task, "deep_correction", False) or getattr(self.settings, "deep_correction", False))
        prompt_mode = "deep" if deep_mode else "polish"
        if deep_mode:
            self.log_message.emit(
                "info",
                "Deep conversational correction enabled — allowing broader paraphrasing.",
            )
        for chunk in chunked(texts, batch_size, max_tokens=prompt_limit, token_margin=token_margin):
            try:
                self.log_message.emit(
                    "info",
                    f"LM Studio correction: sending {len(chunk)} line(s) (mode={prompt_mode}).",
                )
            except RuntimeError:
                pass
            rewritten.extend(client.rewrite_batch(chunk, lang_hint, mode=prompt_mode))
        adjusted = self._apply_correction_margin(segments, rewritten)
        for seg, new_text in zip(segments, adjusted):
            seg["text"] = new_text
        try:
            self.log_message.emit(
                "success",
                f"LM Studio correction finished: {len(segments)} line(s) updated.",
            )
        except RuntimeError:
            pass
        return segments

    # Подгоняем количество строк ответа LM Studio под исходное число сегментов.
    def _apply_correction_margin(self, segments: List[dict], rewritten: List[str]) -> List[str]:
        expected = len(segments)
        if expected == 0:
            return []
        actual = len(rewritten)
        if actual == expected:
            return rewritten
        tolerance = max(1, int(expected * 0.1))
        if abs(actual - expected) > tolerance:
            self.log_message.emit("warning", "LM Studio returned unexpected number of lines. Keeping original text.")
            return [seg.get("text", "") for seg in segments]
        self.log_message.emit(
            "info",
            f"LM Studio returned {actual} lines. Adjusting to {expected} within +/-10% tolerance.",
        )
        if actual > expected:
            adjusted: List[str] = []
            for index in range(expected):
                start = math.floor(index * actual / expected)
                end = math.floor((index + 1) * actual / expected)
                if end <= start:
                    end = min(actual, start + 1)
                chunk = rewritten[start:end]
                combined = " ".join(chunk).strip()
                if not combined:
                    combined = segments[index].get("text", "")
                adjusted.append(combined)
            return adjusted
        adjusted = list(rewritten)
        adjusted.extend(seg.get("text", "") for seg in segments[actual:])
        return adjusted

    # Полный цикл обработки задачи: подготовка аудио, транскрибация, пост-обработка и очистка.
    def _process_task_for_context(self, context: _DeviceContext, task: TranscriptionTask):
        audio_path: Optional[Path] = None
        cleanup = False
        self._increment_active_transcriptions()
        try:
            self._ensure_ffmpeg_available()
            self.log_message.emit("info", f"Starting transcription: {task.video_path.name}")
            self._mark_task_start(task)
            self.progress_updated.emit(task.task_id, 5)
            if not self._ensure_model_for_context(
                context,
                task.whisper_backend or "openai",
                task.model_size,
                task.faster_whisper_compute_type or "int8",
            ):
                if context.device == "cuda" and "cpu" in self._contexts and context is not self._contexts["cpu"]:
                    self.log_message.emit("warning", f"Falling back to CPU for {task.video_path.name}")
                    task.device = "cpu"
                    self._enqueue_task_for_context(self._contexts["cpu"], task)
                    return
                raise RuntimeError("Unable to load transcription model.")
            self.progress_updated.emit(task.task_id, 15)
            audio_path, cleanup = self._extract_audio(task.video_path)
            self.progress_updated.emit(task.task_id, 30)
            self.log_message.emit("info", f"Processing audio {audio_path.name}...")
            try:
                segments = self._transcribe(context, audio_path, task.language)
            except Exception as exc:
                message = str(exc)
                if (
                    context.runtime_device == "cuda"
                    and self._is_cuda_dependency_error(message)
                ):
                    global _CUDA_PATH_LOGGED
                    if not _CUDA_PATH_LOGGED:
                        path_info = "; ".join(
                            [
                                "DLL search paths: " + ", ".join(_CUDA_DLL_PATHS or ["<none>"]),
                                "PATH env: " + os.environ.get("PATH", ""),
                            ]
                        )
                        self.log_message.emit("warning", path_info)
                        _CUDA_PATH_LOGGED = True
                    self.log_message.emit(
                        "warning",
                        f"CUDA runtime error during transcription: {exc.__class__.__name__}: {message}",
                    )
                    cpu_context = self._contexts.get("cpu")
                    if cpu_context and cpu_context is not context:
                        self.log_message.emit("info", "Retrying transcription on CPU...")
                        task.device = "cpu"
                        self._release_context_model(context)
                        self._enqueue_task_for_context(cpu_context, task)
                        return
                    self.log_message.emit(
                        "error",
                        "CPU fallback unavailable. Please reinstall CUDA dependencies or switch device to CPU.",
                    )
                if (
                    (context.backend or "").lower() == "faster"
                    and self._is_feature_shape_error(message)
                    and not task.backend_fallback_attempted
                ):
                    self.log_message.emit(
                        "warning",
                        f"Faster-Whisper feature-shape error: {message}. Falling back to openai-whisper.",
                    )
                    task.whisper_backend = "openai"
                    task.backend_fallback_attempted = True
                    self._release_context_model(context)
                    self._enqueue_task_for_context(context, task)
                    return
                if (
                    (context.backend or "").lower() == "faster"
                    and self._is_feature_shape_error(message)
                    and task.backend_fallback_attempted
                ):
                    self.log_message.emit(
                        "error",
                        f"Faster-Whisper feature-shape error persists after fallback: {message}",
                    )
                raise
            segments = self._apply_speaker_diarization(task, audio_path, segments)
            self.progress_updated.emit(task.task_id, 70)
            self._handle_post_transcription(task, segments)
        except Exception as exc:
            self.task_failed.emit(task.task_id, str(exc))
            self.log_message.emit("error", f"Transcription error for {task.video_path.name}: {exc}")
            self._log_task_timing(task, "failed")
            self._notify_correction_done(task.task_id)
        finally:
            if cleanup and audio_path and audio_path.exists():
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass

    # Помещаем задачу в очередь конкретного устройства, запуская поток при необходимости.
    def _enqueue_task_for_context(self, context: _DeviceContext, task: TranscriptionTask):
        self._start_context_thread(context)
        context.queue.put(task)

    # Запускаем рабочий поток для устройства, если он ещё не активен.
    def _start_context_thread(self, context: _DeviceContext):
        if context.thread and context.thread.is_alive():
            return
        context.stop_event.clear()
        context.thread = threading.Thread(target=self._device_worker_loop, args=(context,), daemon=True)
        context.thread.start()

    # Выбираем подходящий контекст устройства исходя из настроек задачи.
    def _choose_context_for_task(self, task: TranscriptionTask) -> _DeviceContext:
        device = (task.device or "cpu").lower()
        if device == "auto":
            device = "cuda" if "cuda" in self._contexts else "cpu"
        if device == "hybrid":
            device = self._select_hybrid_device()
        if device not in self._contexts:
            device = "cpu"
        return self._contexts[device]

    # В гибридном режиме балансируем задачи между GPU и CPU.
    def _select_hybrid_device(self) -> str:
        gpu_ctx = self._contexts.get("cuda")
        cpu_ctx = self._contexts.get("cpu")
        if gpu_ctx and cpu_ctx:
            if not gpu_ctx.active:
                self._hybrid_last_device = "cuda"
                return "cuda"
            if not cpu_ctx.active:
                self._hybrid_last_device = "cpu"
                return "cpu"
            gpu_size = gpu_ctx.queue.qsize()
            cpu_size = cpu_ctx.queue.qsize()
            if gpu_size == cpu_size:
                self._hybrid_last_device = "cpu" if self._hybrid_last_device == "cuda" else "cuda"
                return self._hybrid_last_device
            if gpu_size < cpu_size:
                self._hybrid_last_device = "cuda"
                return "cuda"
            self._hybrid_last_device = "cpu"
            return "cpu"
        if gpu_ctx:
            self._hybrid_last_device = "cuda"
            return "cuda"
        self._hybrid_last_device = "cpu"
        return "cpu"

    # Основной цикл работы устройства: берём задачи из очереди и обрабатываем их.
    def _device_worker_loop(self, context: _DeviceContext):
        while not self._shutdown_event.is_set() and not context.stop_event.is_set():
            try:
                task = context.queue.get(timeout=0.1)
            except Empty:
                if (
                    self.settings.release_whisper_after_batch
                    and not context.active
                    and context.queue.empty()
                ):
                    self._release_context_model(context)
                continue
            if task is None:
                context.queue.task_done()
                break
            while self._is_processing_paused and not self._shutdown_event.is_set():
                time.sleep(0.1)
            if self._shutdown_event.is_set():
                context.queue.task_done()
                break
            context.active = True
            try:
                self._process_task_for_context(context, task)
            finally:
                context.active = False
                self._decrement_active_transcriptions()
                context.queue.task_done()
            if self.settings.release_whisper_after_batch and context.queue.empty():
                self._release_context_model(context)

    # Главный цикл QThread: распределяет задачи по устройствам и управляет завершением.
    def run(self):
        while self._is_running:
            if self._is_processing_paused:
                self.msleep(150)
                continue
            try:
                task = self.tasks_queue.get(timeout=0.05)
            except Empty:
                self.msleep(50)
                continue
            try:
                context = self._choose_context_for_task(task)
                self._enqueue_task_for_context(context, task)
            finally:
                self.tasks_queue.task_done()
        self._shutdown_event.set()
        for context in self._contexts.values():
            context.stop_event.set()
            context.queue.put(None)
        for context in self._contexts.values():
            if context.thread and context.thread.is_alive():
                context.thread.join()
        self._shutdown_correction_worker()
        if self._correction_thread.is_alive():
            self._correction_thread.join()
        self._release_all_models()

    # Инициируем остановку рабочего потока и связанных очередей.
    def stop(self):
        self._is_running = False
        self._shutdown_event.set()
        self.clear_queue()
        self._shutdown_correction_worker()
        for context in self._contexts.values():
            context.stop_event.set()
            context.queue.put(None)

    # Проверяем наличие ffmpeg в PATH один раз за запуск; без него транскрибация не стартует.
    def _ensure_ffmpeg_available(self):
        if self._ffmpeg_checked:
            if not self._ffmpeg_available:
                raise RuntimeError("FFmpeg executable is required but not available.")
            return
        self._ffmpeg_checked = True
        self._ffmpeg_available = shutil.which("ffmpeg") is not None
        if not self._ffmpeg_available:
            message = (
                "FFmpeg executable not found. Install FFmpeg and ensure it is available in PATH "
                "before running transcription."
            )
            self.log_message.emit("error", message)
            raise RuntimeError(message)
