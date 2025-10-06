import time
from pathlib import Path
from queue import Queue, Empty

from PyQt6.QtCore import QThread, pyqtSignal
import srt

from .g4f_client import g4f_batch_translate


class TranslationTask(tuple):
    __slots__ = ()
    _fields = ('task_id', 'source_path', 'target_lang', 'use_g4f', 'g4f_model', 'source_lang')

    def __new__(cls, task_id: str, source_path: Path, target_lang: str, use_g4f: bool, g4f_model: str,
                source_lang: str = "auto"):
        return tuple.__new__(cls, (task_id, source_path, target_lang, use_g4f, g4f_model, source_lang))

    @property
    def task_id(self): return self[0]

    @property
    def source_path(self): return self[1]

    @property
    def target_lang(self): return self[2]

    @property
    def use_g4f(self): return self[3]

    @property
    def g4f_model(self): return self[4]

    @property
    def source_lang(self): return self[5]


class TranslationWorker(QThread):
    translation_completed = pyqtSignal(str, str)
    translation_failed = pyqtSignal(str, str)
    log_message = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.tasks_queue = Queue()
        self._is_running = True

    def add_task(self, task: TranslationTask):
        self.tasks_queue.put(task)

    def _translate_srt_g4f(self, source_path: Path, target_lang: str, g4f_model: str, source_lang: str) -> Path:
        with open(source_path, 'r', encoding='utf-8') as f:
            subs = list(srt.parse(f.read()))
        batch_size = 25
        contents = [s.content for s in subs]
        translated = []
        for i in range(0, len(contents), batch_size):
            chunk = contents[i:i + batch_size]
            out = g4f_batch_translate(chunk, g4f_model, target_lang, source_lang)
            translated.extend(out)
            time.sleep(0.2)
        for s, t in zip(subs, translated):
            s.content = t
        output_path = source_path.with_name(f"{source_path.stem}_{target_lang}.srt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(subs))
        return output_path

    def _translate_txt_g4f(self, source_path: Path, target_lang: str, g4f_model: str, source_lang: str) -> Path:
        with open(source_path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        batch_size = 50
        translated = []
        for i in range(0, len(lines), batch_size):
            chunk = lines[i:i + batch_size]
            out = g4f_batch_translate(chunk, g4f_model, target_lang, source_lang)
            translated.extend(out)
            time.sleep(0.2)
        output_path = source_path.with_name(f"{source_path.stem}_{target_lang}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(translated))
        return output_path

    def _process_task(self, task: TranslationTask):
        try:
            self.log_message.emit("info", f"Начало перевода {task.source_path.name} на '{task.target_lang}'")
            if task.source_path.suffix.lower() == '.srt':
                result_path = self._translate_srt_g4f(task.source_path, task.target_lang, task.g4f_model,
                                                      task.source_lang)
            elif task.source_path.suffix.lower() == '.txt':
                result_path = self._translate_txt_g4f(task.source_path, task.target_lang, task.g4f_model,
                                                      task.source_lang)
            else:
                raise ValueError(f"Неподдерживаемый формат: {task.source_path.suffix}")
            self.log_message.emit("success", f"Перевод завершён: {Path(result_path).name}")
            self.translation_completed.emit(task.task_id, str(result_path))
        except Exception as e:
            self.log_message.emit("error", f"Ошибка перевода: {e}")
            self.translation_failed.emit(task.task_id, str(e))

    def run(self):
        while self._is_running:
            try:
                task = self.tasks_queue.get(timeout=0.1)
                self._process_task(task)
                self.tasks_queue.task_done()
            except Empty:
                self.msleep(100)

    def stop(self):
        self._is_running = False
