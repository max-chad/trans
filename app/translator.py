from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal
import srt

from .lmstudio_client import LmStudioClient, LmStudioError, LmStudioSettings, chunked


@dataclass
class TranslationTask:
    task_id: str
    source_path: Path
    target_lang: str
    source_lang: str = "auto"
    use_lmstudio: bool = True
    lmstudio_base_url: str = ""
    lmstudio_model: str = ""
    lmstudio_api_key: str = ""
    lmstudio_batch_size: int = 40
    lmstudio_prompt_token_limit: int = 8192
    lmstudio_response_token_limit: int = 4096
    lmstudio_token_margin: int = 512
    lmstudio_load_timeout: float = 600.0
    lmstudio_poll_interval: float = 1.5


class TranslationWorker(QThread):
    translation_completed = pyqtSignal(str, str)
    translation_failed = pyqtSignal(str, str)
    log_message = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.tasks_queue: "Queue[TranslationTask]" = Queue()
        self._is_running = True

    def add_task(self, task: TranslationTask):
        self.tasks_queue.put(task)

    def _build_client(self, task: TranslationTask) -> LmStudioClient:
        base_url = (task.lmstudio_base_url or "").strip()
        model = (task.lmstudio_model or "").strip()
        if not base_url or not model:
            raise ValueError("LM Studio settings are incomplete.")
        settings = LmStudioSettings(
            base_url=base_url,
            model=model,
            api_key=task.lmstudio_api_key or "",
            timeout=120.0,
            temperature=0.2,
            max_completion_tokens=task.lmstudio_response_token_limit or 4096,
            max_prompt_tokens=task.lmstudio_prompt_token_limit or 8192,
            prompt_token_margin=max(0, task.lmstudio_token_margin),
        )
        return LmStudioClient(settings)

    def _translate_srt(
        self,
        task: TranslationTask,
        client: LmStudioClient,
    ) -> Path:
        with open(task.source_path, "r", encoding="utf-8") as f:
            subtitles = list(srt.parse(f.read()))
        texts = [sub.content for sub in subtitles]
        translated: list[str] = []
        batch_size = max(1, task.lmstudio_batch_size)
        prompt_limit = max(0, task.lmstudio_prompt_token_limit)
        token_margin = max(0, task.lmstudio_token_margin)
        for chunk in chunked(texts, batch_size, max_tokens=prompt_limit, token_margin=token_margin):
            translated.extend(
                client.translate_batch(chunk, task.target_lang, task.source_lang)
            )
        if len(translated) != len(subtitles):
            raise LmStudioError(
                f"Ожидалось {len(subtitles)} строк, LM Studio вернул {len(translated)}."
            )
        for sub, text in zip(subtitles, translated):
            sub.content = text
        output_path = task.source_path.with_name(f"{task.source_path.stem}_{task.target_lang}.srt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt.compose(subtitles))
        return output_path

    def _translate_txt(
        self,
        task: TranslationTask,
        client: LmStudioClient,
    ) -> Path:
        with open(task.source_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        batch_size = max(1, task.lmstudio_batch_size)
        translated: list[str] = []
        prompt_limit = max(0, task.lmstudio_prompt_token_limit)
        token_margin = max(0, task.lmstudio_token_margin)
        for chunk in chunked(lines, batch_size, max_tokens=prompt_limit, token_margin=token_margin):
            translated.extend(
                client.translate_batch(chunk, task.target_lang, task.source_lang)
            )
        output_path = task.source_path.with_name(f"{task.source_path.stem}_{task.target_lang}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(translated))
        return output_path

    def _process_task(self, task: TranslationTask):
        try:
            if not task.use_lmstudio:
                raise ValueError("LM Studio отключён. Перевод невозможен.")
            client = self._build_client(task)
            load_timeout = float(getattr(task, 'lmstudio_load_timeout', 0) or 0)
            poll_interval = float(getattr(task, 'lmstudio_poll_interval', 0) or 0)
            try:
                client.ensure_model_loaded(
                    lambda level, message: self.log_message.emit(level, message),
                    timeout=load_timeout if load_timeout else 600.0,
                    poll_interval=poll_interval if poll_interval else 1.5,
                )
            except LmStudioError as exc:
                raise ValueError(f"LM Studio модель не готова: {exc}") from exc
            if task.source_path.suffix.lower() == ".srt":
                result_path = self._translate_srt(task, client)
            elif task.source_path.suffix.lower() == ".txt":
                result_path = self._translate_txt(task, client)
            else:
                raise ValueError(f"Формат файла не поддерживается: {task.source_path.suffix}")
            self.log_message.emit(
                "success", f"Перевод завершён: {Path(result_path).name}"
            )
            self.translation_completed.emit(task.task_id, str(result_path))
        except (LmStudioError, ValueError) as exc:
            self.log_message.emit("error", f"Ошибка перевода: {exc}")
            self.translation_failed.emit(task.task_id, str(exc))

    def run(self):
        while self._is_running:
            try:
                task = self.tasks_queue.get(timeout=0.1)
            except Empty:
                self.msleep(100)
                continue
            self._process_task(task)
            self.tasks_queue.task_done()

    def stop(self):
        self._is_running = False
