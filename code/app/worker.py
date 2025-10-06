import os
from pathlib import Path
from typing import List, Optional
from queue import Queue, Empty
from datetime import timedelta

from PyQt6.QtCore import QThread, pyqtSignal
import whisper
import torch
from moviepy.editor import VideoFileClip
import srt

from .models import TranscriptionTask, DeviceType
from .g4f_client import g4f_batch_rewrite


class TranscriptionWorker(QThread):
    progress_updated = pyqtSignal(str, int)
    task_completed = pyqtSignal(str, str)
    task_failed = pyqtSignal(str, str)
    log_message = pyqtSignal(str, str)
    model_loaded = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.tasks_queue = Queue()
        self.current_model = None
        self.current_model_size: Optional[str] = None
        self.current_device: Optional[str] = None
        self._is_running = True
        self._is_processing_paused = False

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

    def _load_model(self, size: str, device: str):
        if self.current_model and self.current_model_size == size and self.current_device == device:
            return
        self.current_model = None
        self.current_model_size = size
        self.current_device = device
        resolved_device = "cuda" if device == DeviceType.CUDA.value and torch.cuda.is_available() else "cpu"
        self.log_message.emit("info", f"Загрузка модели Whisper '{size}' на '{resolved_device}'...")
        try:
            self.current_model = whisper.load_model(size, device=resolved_device)
            self.log_message.emit("success", "Модель загружена.")
            self.model_loaded.emit(True)
        except RuntimeError as e:
            if "CUDA" in str(e) and resolved_device == "cuda":
                self.log_message.emit("warning", f"Ошибка CUDA: {e}")
                self.log_message.emit("info", "Переключение на CPU...")
                try:
                    self.current_model = whisper.load_model(size, device="cpu")
                    self.current_device = "cpu"
                    self.log_message.emit("success", "Модель загружена на CPU.")
                    self.model_loaded.emit(True)
                except Exception as cpu_error:
                    self.log_message.emit("error", f"Ошибка загрузки на CPU: {cpu_error}")
                    self.model_loaded.emit(False)
            else:
                self.log_message.emit("error", f"Ошибка загрузки модели: {e}")
                self.model_loaded.emit(False)
        except Exception as e:
            self.log_message.emit("error", f"Ошибка загрузки модели: {e}")
            self.model_loaded.emit(False)

    def _extract_audio(self, video_path: Path) -> Path:
        self.log_message.emit("info", f"Извлечение аудио из {video_path.name}...")
        audio_path = video_path.with_name(f"{video_path.stem}_temp_audio.wav")
        video = VideoFileClip(str(video_path))
        video.audio.write_audiofile(str(audio_path), codec='pcm_s16le', logger=None)
        video.close()
        return audio_path

    def _save_as_srt(self, segments: List[dict], output_path: Path):
        srt_segments = [
            srt.Subtitle(
                index=i,
                start=timedelta(seconds=seg['start']),
                end=timedelta(seconds=seg['end']),
                content=seg['text'].strip()
            )
            for i, seg in enumerate(segments, 1)
        ]
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(srt_segments))

    def _save_as_txt(self, segments: List[dict], output_path: Path):
        with open(output_path, 'w', encoding='utf-8') as f:
            for segment in segments:
                f.write(segment['text'].strip() + '\n')

    def _g4f_refine_segments(self, segments: List[dict], model: str, lang_hint: str):
        texts = [seg['text'] for seg in segments]
        batch_size = 40
        refined = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            out = g4f_batch_rewrite(chunk, model, lang_hint)
            refined.extend(out)
        for seg, new_text in zip(segments, refined):
            seg['text'] = new_text

    def _process_task(self, task: TranscriptionTask):
        try:
            self.log_message.emit("info", f"Начало задачи для: {task.video_path.name}")
            self.progress_updated.emit(task.task_id, 5)
            self._load_model(task.model_size, task.device)
            if not self.current_model:
                raise RuntimeError("Модель не загружена.")
            self.progress_updated.emit(task.task_id, 15)
            audio_path = self._extract_audio(task.video_path)
            self.progress_updated.emit(task.task_id, 30)
            self.log_message.emit("info", f"Транскрибация аудио для {task.video_path.name}...")
            result = self.current_model.transcribe(
                str(audio_path),
                language=task.language if task.language != "auto" else None,
                fp16=torch.cuda.is_available() and self.current_device == "cuda",
                verbose=False
            )
            self.progress_updated.emit(task.task_id, 70)
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            segments = result.get('segments', [])
            if task.use_g4f_correction and segments:
                self.log_message.emit("info", "Коррекция текста через g4f...")
                self._g4f_refine_segments(segments, task.g4f_model, task.language if task.language != "auto" else "")
            self.progress_updated.emit(task.task_id, 85)
            output_name = task.video_path.stem
            output_path = task.output_dir / f"{output_name}.{task.output_format}"
            if task.output_format == "srt":
                self._save_as_srt(segments, output_path)
            else:
                self._save_as_txt(segments, output_path)
            self.progress_updated.emit(task.task_id, 100)
            self.task_completed.emit(task.task_id, str(output_path))
            self.log_message.emit("success", f"Задача завершена для: {task.video_path.name}")
        except Exception as e:
            self.task_failed.emit(task.task_id, str(e))
            self.log_message.emit("error", f"Ошибка задачи для {task.video_path.name}: {e}")

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
