import json
import os
from datetime import timedelta
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, List, Optional, TYPE_CHECKING

import srt
from PyQt6.QtCore import QThread, pyqtSignal
from moviepy.editor import VideoFileClip

from asr.transcriber import ASRTranscriber
from nlp.local_corrector import LocalTextCorrector
from utils.memory import free_cuda

try:  # pragma: no cover - optional GPU dependency
    import torch
except Exception:  # pragma: no cover
    torch = None

from .models import DeviceType, TranscriptionTask

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    from .config import AppConfig


class TranscriptionWorker(QThread):
    progress_updated = pyqtSignal(str, int)
    task_completed = pyqtSignal(str, str)
    task_failed = pyqtSignal(str, str)
    log_message = pyqtSignal(str, str)

    def __init__(self, app_config: "AppConfig"):
        super().__init__()
        self.config = app_config
        self.tasks_queue: "Queue[TranscriptionTask]" = Queue()
        self._is_running = True
        self._is_processing_paused = False

    # ------------------------------------------------------------------
    def add_task(self, task: TranscriptionTask):
        self.tasks_queue.put(task)

    def stop_processing(self):
        self._is_processing_paused = True
        self.clear_queue()

    def resume_processing(self):
        self._is_processing_paused = False

    def clear_queue(self):
        with self.tasks_queue.mutex:
            self.tasks_queue.queue.clear()

    # ------------------------------------------------------------------
    def _extract_audio(self, video_path: Path) -> Path:
        self.log_message.emit("info", f"Извлечение аудио из {video_path.name}...")
        audio_path = video_path.with_name(f"{video_path.stem}_temp_audio.wav")
        clip = VideoFileClip(str(video_path))
        clip.audio.write_audiofile(str(audio_path), codec="pcm_s16le", logger=None)
        clip.close()
        return audio_path

    @staticmethod
    def _load_segments(path: Path) -> List[Dict[str, object]]:
        segments: List[Dict[str, object]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    segments.append(json.loads(line))
        return segments

    @staticmethod
    def _write_segments(path: Path, segments: List[Dict[str, object]]):
        with path.open("w", encoding="utf-8") as f:
            for seg in segments:
                f.write(json.dumps(seg, ensure_ascii=False) + "\n")

    def _save_as_srt(self, segments: List[Dict[str, object]], output_path: Path):
        srt_segments = [
            srt.Subtitle(
                index=i,
                start=timedelta(seconds=float(seg.get("start", 0.0))),
                end=timedelta(seconds=float(seg.get("end", 0.0))),
                content=str(seg.get("text", "")).strip(),
            )
            for i, seg in enumerate(segments, start=1)
        ]
        with output_path.open("w", encoding="utf-8") as f:
            f.write(srt.compose(srt_segments))

    def _save_as_txt(self, segments: List[Dict[str, object]], output_path: Path):
        with output_path.open("w", encoding="utf-8") as f:
            for seg in segments:
                f.write(str(seg.get("text", "")).strip() + "\n")

    # ------------------------------------------------------------------
    def _process_task(self, task: TranscriptionTask):
        try:
            self.log_message.emit("info", f"Начало задачи для: {task.video_path.name}")
            self.progress_updated.emit(task.task_id, 5)

            asr_cfg = self.config.build_asr_config()
            llm_cfg = self.config.build_llm_config()
            runtime_flags = self.config.runtime_flags()
            allow_remote = bool(runtime_flags.get("allow_remote") and task.allow_remote)

            asr_cfg["model"] = task.model_size
            target_device = DeviceType.CUDA.value if task.device == DeviceType.CUDA.value else DeviceType.CPU.value
            if target_device == DeviceType.CUDA.value and (torch is None or not torch.cuda.is_available()):
                self.log_message.emit("warning", "CUDA недоступна, переключение на CPU")
                target_device = DeviceType.CPU.value
            asr_cfg["device"] = target_device
            asr_cfg["compute_type"] = "float16" if target_device == DeviceType.CUDA.value else "int8"

            temp_dir = self.config.temp_directory()
            temp_dir.mkdir(parents=True, exist_ok=True)

            raw_path = temp_dir / f"{task.video_path.stem}.raw.jsonl"
            clean_path = temp_dir / f"{task.video_path.stem}.clean.jsonl"

            audio_path = self._extract_audio(task.video_path)
            self.progress_updated.emit(task.task_id, 25)

            self.log_message.emit("info", f"Stage A: транскрибация {task.video_path.name}")
            with ASRTranscriber(asr_cfg, allow_remote=allow_remote) as transcriber:
                transcriber.transcribe(audio_path, raw_path)
            self.progress_updated.emit(task.task_id, 55)
            free_cuda()

            if audio_path.exists():
                os.unlink(audio_path)

            segments = self._load_segments(raw_path)
            batch_size = max(1, int(llm_cfg.get("batch_size", 4)))
            corrected: List[Dict[str, object]] = []
            self.log_message.emit("info", "Stage B: корректировка текста")
            with LocalTextCorrector(llm_cfg, allow_remote=allow_remote) as corrector:
                for i in range(0, len(segments), batch_size):
                    batch = [str(seg.get("text", "")) for seg in segments[i : i + batch_size]]
                    updated = corrector.correct_batch(batch)
                    for seg, new_text in zip(segments[i : i + batch_size], updated):
                        corrected.append({**seg, "text": new_text})
            self._write_segments(clean_path, corrected)
            self.progress_updated.emit(task.task_id, 75)
            free_cuda()

            self.log_message.emit("info", "Stage C: экспорт результатов")
            output_path = task.output_dir / f"{task.video_path.stem}.{task.output_format}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if task.output_format.lower() == "srt":
                self._save_as_srt(corrected, output_path)
            else:
                self._save_as_txt(corrected, output_path)
            self.progress_updated.emit(task.task_id, 100)

            task.result_path = output_path
            self.task_completed.emit(task.task_id, str(output_path))
            self.log_message.emit("success", f"Задача завершена для: {task.video_path.name}")
        except Exception as exc:  # pragma: no cover - GUI feedback
            self.task_failed.emit(task.task_id, str(exc))
            self.log_message.emit("error", f"Ошибка задачи для {task.video_path.name}: {exc}")
        finally:
            free_cuda()

    # ------------------------------------------------------------------
    def run(self):
        while self._is_running:
            if not self._is_processing_paused:
                try:
                    task = self.tasks_queue.get(timeout=0.1)
                    self._process_task(task)
                    self.tasks_queue.task_done()
                except Empty:
                    self.msleep(100)
            else:
                self.msleep(200)

    def stop(self):
        self._is_running = False
        self.clear_queue()
