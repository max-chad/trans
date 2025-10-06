from pathlib import Path
import time

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import torch

from app.models import TranscriptionTask, DeviceType
from app.worker import TranscriptionWorker
from app.translator import TranslationWorker, TranslationTask
from app.config import AppConfig
from .task_widget import VideoTaskWidget
from .styles import AppTheme


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.tasks = {}
        self.task_widgets = {}
        self.worker = TranscriptionWorker()
        self.worker.progress_updated.connect(self.on_progress_updated)
        self.worker.task_completed.connect(self.on_task_completed)
        self.worker.task_failed.connect(self.on_task_failed)
        self.worker.log_message.connect(self.log_message)
        self.worker.start()
        self.translator = TranslationWorker()
        self.translator.translation_completed.connect(self.on_translation_completed)
        self.translator.translation_failed.connect(self.on_translation_failed)
        self.translator.log_message.connect(self.log_message)
        self.translator.start()
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        self.setWindowTitle("Video Transcriber")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(AppTheme.GLOBAL_STYLE)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, 3)
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, 2)

    def _create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(20)
        self.drop_area = self._create_drop_area()
        layout.addWidget(self.drop_area)
        controls = self._create_controls_panel()
        layout.addWidget(controls)
        tasks_header_layout = QHBoxLayout()
        tasks_header_label = QLabel("Очередь задач")
        tasks_header_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        tasks_header_layout.addWidget(tasks_header_label)
        tasks_header_layout.addStretch()
        clear_all_btn = QPushButton("Очистить все")
        clear_all_btn.setStyleSheet(AppTheme.SECONDARY_BUTTON_STYLE)
        clear_all_btn.clicked.connect(self.clear_all_tasks)
        tasks_header_layout.addWidget(clear_all_btn)
        layout.addLayout(tasks_header_layout)
        self.tasks_scroll = QScrollArea()
        self.tasks_scroll.setWidgetResizable(True)
        self.tasks_scroll.setStyleSheet(f"""
            QScrollArea {{ 
                background-color: {AppTheme.PANELS}; 
                border: 1px solid {AppTheme.BORDER}; 
                border-radius: 12px;
            }}
        """)
        self.tasks_container = QWidget()
        self.tasks_container.setStyleSheet("background-color: transparent;")
        self.tasks_layout = QVBoxLayout(self.tasks_container)
        self.tasks_layout.setSpacing(10)
        self.tasks_layout.setContentsMargins(10, 10, 10, 10)
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.tasks_layout.addItem(spacer)
        self.tasks_scroll.setWidget(self.tasks_container)
        layout.addWidget(self.tasks_scroll)
        return panel

    def _create_drop_area(self):
        drop_area = QLabel("Перетащите видео файлы сюда\nили нажмите для выбора")
        drop_area.setAcceptDrops(True)
        drop_area.setMinimumHeight(150)
        drop_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_area.setStyleSheet(f"""
            QLabel {{
                background-color: {AppTheme.PANELS};
                border: 2px dashed {AppTheme.BORDER};
                border-radius: 15px;
                color: {AppTheme.TEXT_SECONDARY};
                font-size: 16px;
            }}
            QLabel:hover {{
                border-color: {AppTheme.ACCENT};
                color: {AppTheme.TEXT_PRIMARY};
            }}
        """)

        def drag_enter_event(event):
            if event.mimeData().hasUrls():
                event.acceptProposedAction()

        def drop_event(event):
            files = [Path(url.toLocalFile()) for url in event.mimeData().urls()]
            self.add_video_files(files)

        drop_area.dragEnterEvent = drag_enter_event
        drop_area.dropEvent = drop_event
        drop_area.mousePressEvent = lambda _: self.browse_files()
        return drop_area

    def _create_controls_panel(self):
        controls_group = QGroupBox("Настройки")
        controls_group.setStyleSheet(AppTheme.GROUPBOX_STYLE)
        layout = QGridLayout(controls_group)
        layout.addWidget(QLabel("Папка для сохранения:"), 0, 0)
        self.output_label = QLabel("Не выбрана")
        self.output_label.setStyleSheet(
            f"color: {AppTheme.TEXT_SECONDARY}; padding: 5px; border: 1px solid {AppTheme.BORDER}; border-radius: 5px;")
        layout.addWidget(self.output_label, 0, 1)
        output_btn = QPushButton("Обзор")
        output_btn.setStyleSheet(AppTheme.SECONDARY_BUTTON_STYLE)
        output_btn.clicked.connect(self.select_output_dir)
        layout.addWidget(output_btn, 0, 2)
        layout.addWidget(QLabel("Язык транскрибации:"), 1, 0)
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["auto", "ru", "en", "de", "fr", "es", "it", "uk", "pl"])
        self.lang_combo.setStyleSheet(AppTheme.COMBOBOX_STYLE)
        layout.addWidget(self.lang_combo, 1, 1, 1, 2)
        layout.addWidget(QLabel("Модель:"), 2, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setStyleSheet(AppTheme.COMBOBOX_STYLE)
        layout.addWidget(self.model_combo, 2, 1, 1, 2)
        layout.addWidget(QLabel("Устройство:"), 3, 0)
        self.device_group = QButtonGroup()
        device_layout = QHBoxLayout()
        self.cpu_radio = QRadioButton("CPU")
        self.cpu_radio.setStyleSheet(AppTheme.RADIOBUTTON_STYLE)
        self.gpu_radio = QRadioButton("GPU (CUDA)")
        self.gpu_radio.setStyleSheet(AppTheme.RADIOBUTTON_STYLE)
        self.gpu_radio.setEnabled(torch.cuda.is_available())
        self.device_group.addButton(self.cpu_radio)
        self.device_group.addButton(self.gpu_radio)
        device_layout.addWidget(self.cpu_radio)
        device_layout.addWidget(self.gpu_radio)
        layout.addLayout(device_layout, 3, 1, 1, 2)
        layout.addWidget(QLabel("Формат:"), 4, 0)
        self.format_group = QButtonGroup()
        format_layout = QHBoxLayout()
        self.srt_radio = QRadioButton("SRT")
        self.srt_radio.setStyleSheet(AppTheme.RADIOBUTTON_STYLE)
        self.txt_radio = QRadioButton("TXT")
        self.txt_radio.setStyleSheet(AppTheme.RADIOBUTTON_STYLE)
        self.format_group.addButton(self.srt_radio)
        self.format_group.addButton(self.txt_radio)
        format_layout.addWidget(self.srt_radio)
        format_layout.addWidget(self.txt_radio)
        layout.addLayout(format_layout, 4, 1, 1, 2)
        layout.addWidget(QLabel("Перевести на язык:"), 5, 0)
        self.translate_lang_combo = QComboBox()
        self.translate_lang_combo.addItems(["en", "ru", "de", "fr", "es", "it", "uk", "pl"])
        self.translate_lang_combo.setStyleSheet(AppTheme.COMBOBOX_STYLE)
        layout.addWidget(self.translate_lang_combo, 5, 1, 1, 2)
        return controls_group

    def _create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(20)
        log_header = QLabel("Журнал событий")
        log_header.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(log_header)
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setStyleSheet(f"""
            QTextEdit {{
                background-color: {AppTheme.PANELS};
                border: 1px solid {AppTheme.BORDER};
                border-radius: 12px;
                padding: 15px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 13px;
            }}
        """)
        layout.addWidget(self.log_widget)
        buttons_layout = QHBoxLayout()
        self.process_btn = QPushButton("Начать обработку")
        self.process_btn.setStyleSheet(AppTheme.MAIN_BUTTON_STYLE)
        self.process_btn.clicked.connect(self.start_processing)
        self.stop_btn = QPushButton("Остановить")
        self.stop_btn.setStyleSheet(AppTheme.SECONDARY_BUTTON_STYLE.replace(AppTheme.BORDER, AppTheme.ERROR))
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        buttons_layout.addWidget(self.process_btn)
        buttons_layout.addWidget(self.stop_btn)
        layout.addLayout(buttons_layout)
        return panel

    def load_settings(self):
        self.output_label.setText(self.config.get("output_dir"))
        self.model_combo.setCurrentText(self.config.get("model_size"))
        self.lang_combo.setCurrentText(self.config.get("language"))
        self.translate_lang_combo.setCurrentText(self.config.get("translate_lang") or "en")
        if self.config.get("device") == "cuda":
            self.gpu_radio.setChecked(True)
        else:
            self.cpu_radio.setChecked(True)
        if self.config.get("output_format") == "txt":
            self.txt_radio.setChecked(True)
        else:
            self.srt_radio.setChecked(True)

    def save_settings(self):
        self.config.set("output_dir", self.output_label.text())
        self.config.set("model_size", self.model_combo.currentText())
        self.config.set("language", self.lang_combo.currentText())
        self.config.set("translate_lang", self.translate_lang_combo.currentText())
        self.config.set("device", "cuda" if self.gpu_radio.isChecked() else "cpu")
        self.config.set("output_format", "txt" if self.txt_radio.isChecked() else "srt")

    def browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Выберите видео файлы", "",
                                                "Видео файлы (*.mp4 *.mkv *.avi *.mov)")
        if files:
            self.add_video_files([Path(f) for f in files])

    def select_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения")
        if directory:
            self.output_label.setText(directory)

    def add_video_files(self, file_paths: list[Path]):
        for path in file_paths:
            if any(path.name == task.video_path.name for task in self.tasks.values()):
                self.log_message("warning", f"Файл '{path.name}' уже находится в очереди.")
                continue
            task = TranscriptionTask(video_path=path, output_dir=Path(), output_format="", language="", model_size="")
            self.tasks[task.task_id] = task
            self.add_task_widget(task)

    def add_task_widget(self, task):
        widget = VideoTaskWidget(task)
        widget.remove_requested.connect(self.remove_task)
        widget.translate_requested.connect(self.handle_translation_request)
        self.task_widgets[task.task_id] = widget
        count = self.tasks_layout.count()
        self.tasks_layout.insertWidget(count - 1, widget)

    def remove_task(self, task_id):
        if task_id in self.task_widgets:
            self.task_widgets[task_id].deleteLater()
            del self.task_widgets[task_id]
        if task_id in self.tasks:
            del self.tasks[task_id]

    def clear_all_tasks(self):
        for task_id in list(self.tasks.keys()):
            self.remove_task(task_id)

    def start_processing(self):
        if not self.tasks:
            QMessageBox.warning(self, "Нет задач", "Пожалуйста, добавьте видео файлы для обработки.")
            return
        self.save_settings()
        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker.resume_processing()
        for task_id, task in self.tasks.items():
            if task.status == "pending":
                task.output_dir = Path(self.config.get("output_dir"))
                task.output_format = self.config.get("output_format")
                task.language = self.config.get("language")
                task.model_size = self.config.get("model_size")
                task.device = self.config.get("device")
                task.use_g4f_correction = bool(self.config.get("use_g4f_correction"))
                task.g4f_model = self.config.get("g4f_model")
                task.status = "queued"
                self.worker.add_task(task)
        self.log_message("info", f"Запущена обработка {len(self.tasks)} задач.")

    def stop_processing(self):
        self.log_message("warning", "Обработка всех задач остановлена.")
        self.worker.stop_processing()
        for task in self.tasks.values():
            if task.status == "queued":
                task.status = "pending"
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_progress_updated(self, task_id, progress):
        if task_id in self.task_widgets:
            self.task_widgets[task_id].update_progress(progress)

    def on_task_completed(self, task_id, output_path):
        if task_id in self.tasks:
            self.tasks[task_id].status = "completed"
            self.tasks[task_id].result_path = Path(output_path)
        if task_id in self.task_widgets:
            self.task_widgets[task_id].show_translation_controls()
        self.check_all_tasks_done()

    def on_task_failed(self, task_id, error):
        if task_id in self.task_widgets:
            self.task_widgets[task_id].set_error(error)
        if task_id in self.tasks:
            self.tasks[task_id].status = "failed"
            self.tasks[task_id].error = error
        self.check_all_tasks_done()

    def handle_translation_request(self, task_id: str):
        task = self.tasks.get(task_id)
        if not task or not task.result_path:
            self.log_message("error", "Исходный файл для перевода не найден.")
            return
        self.save_settings()
        target_lang = self.config.get("translate_lang")
        if task.language != "auto" and task.language == target_lang:
            self.log_message("warning", "Исходный язык и язык перевода совпадают.")
            QMessageBox.warning(self, "Перевод", "Исходный язык и язык перевода совпадают.")
            return
        widget = self.task_widgets.get(task_id)
        if widget:
            widget.set_status_translating()
        translation_task = TranslationTask(
            task_id=task_id,
            source_path=task.result_path,
            target_lang=target_lang,
            use_g4f=bool(self.config.get("use_g4f_translation")),
            g4f_model=self.config.get("g4f_model"),
            source_lang=self.config.get("language")
        )
        self.translator.add_task(translation_task)

    def on_translation_completed(self, task_id: str, new_path: str):
        widget = self.task_widgets.get(task_id)
        if widget:
            widget.set_status_translation_complete()

    def on_translation_failed(self, task_id: str, error: str):
        widget = self.task_widgets.get(task_id)
        if widget:
            widget.set_error(f"Ошибка перевода: {error}")

    def check_all_tasks_done(self):
        if all(t.status in ["completed", "failed", "pending"] for t in self.tasks.values()):
            self.process_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    def log_message(self, level, message):
        timestamp = time.strftime("%H:%M:%S")
        color_map = {
            "info": AppTheme.TEXT_SECONDARY,
            "success": AppTheme.SUCCESS,
            "error": AppTheme.ERROR,
            "warning": AppTheme.WARNING
        }
        color = color_map.get(level, AppTheme.TEXT_SECONDARY)
        self.log_widget.append(
            f'<span style="color: {AppTheme.TEXT_SECONDARY};">[{timestamp}]</span> '
            f'<span style="color: {color};">{message}</span>'
        )

    def closeEvent(self, event):
        self.save_settings()
        self.worker.stop()
        self.translator.stop()
        self.worker.wait()
        self.translator.wait()
        event.accept()
