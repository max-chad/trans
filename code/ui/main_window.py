from pathlib import Path
import time

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import QDesktopServices
import torch

from app.models import DeviceType, OutputRequest, ProcessingSettings, TranscriptionTask
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
        self._post_action_triggered = False

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
        controls_group = QGroupBox("Настройки обработки")
        controls_group.setStyleSheet(AppTheme.GROUPBOX_STYLE)
        container_layout = QVBoxLayout(controls_group)
        container_layout.setContentsMargins(12, 12, 12, 12)
        container_layout.setSpacing(12)

        self.settings_tabs = QTabWidget()
        self.settings_tabs.setDocumentMode(True)
        container_layout.addWidget(self.settings_tabs)

        basic_tab = QWidget()
        basic_layout = QGridLayout(basic_tab)
        basic_layout.setColumnStretch(1, 1)

        basic_layout.addWidget(QLabel("Папка результата:"), 0, 0)
        self.output_label = QLabel("")
        self.output_label.setStyleSheet(
            f"color: {AppTheme.TEXT_SECONDARY}; padding: 5px; border: 1px solid {AppTheme.BORDER}; border-radius: 5px;"
        )
        basic_layout.addWidget(self.output_label, 0, 1)
        output_btn = QPushButton("Выбрать")
        output_btn.setStyleSheet(AppTheme.SECONDARY_BUTTON_STYLE)
        output_btn.clicked.connect(self.select_output_dir)
        basic_layout.addWidget(output_btn, 0, 2)

        basic_layout.addWidget(QLabel("Язык распознавания:"), 1, 0)
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["auto", "ru", "en", "de", "fr", "es", "it", "uk", "pl"])
        self.lang_combo.setStyleSheet(AppTheme.COMBOBOX_STYLE)
        basic_layout.addWidget(self.lang_combo, 1, 1, 1, 2)

        basic_layout.addWidget(QLabel("Модель Whisper:"), 2, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setStyleSheet(AppTheme.COMBOBOX_STYLE)
        basic_layout.addWidget(self.model_combo, 2, 1, 1, 2)

        basic_layout.addWidget(QLabel("Устройство инференса:"), 3, 0)
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
        basic_layout.addLayout(device_layout, 3, 1, 1, 2)

        basic_layout.addWidget(QLabel("Форматы субтитров:"), 4, 0)
        formats_layout = QGridLayout()
        self.output_format_checks = {
            "srt": QCheckBox("SRT"),
            "txt_plain": QCheckBox("TXT без таймкодов"),
            "txt_ts": QCheckBox("TXT с таймкодами"),
            "vtt": QCheckBox("VTT"),
        }
        for idx, checkbox in enumerate(self.output_format_checks.values()):
            checkbox.setStyleSheet(AppTheme.RADIOBUTTON_STYLE)
            row = idx // 2
            col = idx % 2
            formats_layout.addWidget(checkbox, row, col)
        basic_layout.addLayout(formats_layout, 4, 1, 1, 2)

        basic_layout.addWidget(QLabel("Язык перевода:"), 5, 0)
        self.translate_lang_combo = QComboBox()
        self.translate_lang_combo.addItems(["en", "ru", "de", "fr", "es", "it", "uk", "pl"])
        self.translate_lang_combo.setStyleSheet(AppTheme.COMBOBOX_STYLE)
        basic_layout.addWidget(self.translate_lang_combo, 5, 1, 1, 2)
        basic_layout.setRowStretch(6, 1)
        self.settings_tabs.addTab(basic_tab, "Основные")

        advanced_tab = QWidget()
        advanced_layout = QGridLayout(advanced_tab)
        advanced_layout.setColumnStretch(1, 1)

        advanced_layout.addWidget(QLabel("Режим обработки:"), 0, 0)
        self.pipeline_mode_combo = QComboBox()
        self.pipeline_mode_combo.addItem("Последовательно", "serial")
        self.pipeline_mode_combo.addItem("Пайплайн (GPU + CPU)", "balanced")
        self.pipeline_mode_combo.addItem("Полностью на GPU (8 ГБ)", "full_gpu")
        self.pipeline_mode_combo.setStyleSheet(AppTheme.COMBOBOX_STYLE)
        advanced_layout.addWidget(self.pipeline_mode_combo, 0, 1, 1, 2)

        advanced_layout.addWidget(QLabel("Потоков транскрибации:"), 1, 0)
        self.transcription_parallel_spin = QSpinBox()
        self.transcription_parallel_spin.setRange(1, 4)
        self.transcription_parallel_spin.setValue(1)
        self.transcription_parallel_spin.setStyleSheet(AppTheme.SPINBOX_STYLE)
        advanced_layout.addWidget(self.transcription_parallel_spin, 1, 1)

        advanced_layout.addWidget(QLabel("Параллельных коррекций:"), 1, 2)
        self.correction_parallel_spin = QSpinBox()
        self.correction_parallel_spin.setRange(1, 8)
        self.correction_parallel_spin.setValue(2)
        self.correction_parallel_spin.setStyleSheet(AppTheme.SPINBOX_STYLE)
        advanced_layout.addWidget(self.correction_parallel_spin, 1, 3)

        advanced_layout.addWidget(QLabel("Коррекция на:"), 2, 0)
        self.correction_device_combo = QComboBox()
        self.correction_device_combo.addItem("Авто", "auto")
        self.correction_device_combo.addItem("CPU", "cpu")
        self.correction_device_combo.addItem("GPU", "gpu")
        self.correction_device_combo.setStyleSheet(AppTheme.COMBOBOX_STYLE)
        advanced_layout.addWidget(self.correction_device_combo, 2, 1, 1, 3)

        advanced_layout.addWidget(QLabel("Строк за вызов LLM:"), 3, 0)
        self.correction_batch_spin = QSpinBox()
        self.correction_batch_spin.setRange(4, 200)
        self.correction_batch_spin.setValue(40)
        self.correction_batch_spin.setStyleSheet(AppTheme.SPINBOX_STYLE)
        advanced_layout.addWidget(self.correction_batch_spin, 3, 1)

        advanced_layout.addWidget(QLabel("Использовать LM Studio:"), 4, 0)
        self.lmstudio_enabled_checkbox = QCheckBox("Включить LM Studio")
        self.lmstudio_enabled_checkbox.setStyleSheet(AppTheme.RADIOBUTTON_STYLE)
        advanced_layout.addWidget(self.lmstudio_enabled_checkbox, 4, 1, 1, 3)

        advanced_layout.addWidget(QLabel("LM Studio URL:"), 5, 0)
        self.lmstudio_base_input = QLineEdit()
        self.lmstudio_base_input.setPlaceholderText("http://127.0.0.1:1234/v1")
        self.lmstudio_base_input.setStyleSheet(
            f"background-color: {AppTheme.PANELS}; border: 1px solid {AppTheme.BORDER}; border-radius: 8px; padding: 8px 12px;"
        )
        advanced_layout.addWidget(self.lmstudio_base_input, 5, 1, 1, 3)

        advanced_layout.addWidget(QLabel("Модель LM Studio:"), 6, 0)
        self.lmstudio_model_input = QLineEdit()
        self.lmstudio_model_input.setPlaceholderText("google/gemma-3n-e4b")
        self.lmstudio_model_input.setStyleSheet(
            f"background-color: {AppTheme.PANELS}; border: 1px solid {AppTheme.BORDER}; border-radius: 8px; padding: 8px 12px;"
        )
        advanced_layout.addWidget(self.lmstudio_model_input, 6, 1, 1, 3)

        advanced_layout.addWidget(QLabel("API Key (если нужен):"), 7, 0)
        self.lmstudio_api_key_input = QLineEdit()
        self.lmstudio_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.lmstudio_api_key_input.setPlaceholderText("Необязательно")
        self.lmstudio_api_key_input.setStyleSheet(
            f"background-color: {AppTheme.PANELS}; border: 1px solid {AppTheme.BORDER}; border-radius: 8px; padding: 8px 12px;"
        )
        advanced_layout.addWidget(self.lmstudio_api_key_input, 7, 1, 1, 3)

        advanced_layout.addWidget(QLabel("Размер пакета LM Studio:"), 8, 0)
        self.lmstudio_batch_spin = QSpinBox()
        self.lmstudio_batch_spin.setRange(1, 200)
        self.lmstudio_batch_spin.setValue(40)
        self.lmstudio_batch_spin.setStyleSheet(AppTheme.SPINBOX_STYLE)
        advanced_layout.addWidget(self.lmstudio_batch_spin, 8, 1)

        advanced_layout.setRowStretch(9, 1)
        self.settings_tabs.addTab(advanced_tab, "Дополнительно")

        self.pipeline_mode_combo.currentIndexChanged.connect(self._update_parallel_controls_state)
        self._update_parallel_controls_state()
        return controls_group
    def _update_parallel_controls_state(self):
        mode = self.pipeline_mode_combo.currentData()
        is_full_gpu = mode == "full_gpu"
        self.transcription_parallel_spin.setEnabled(mode != "serial")
        self.correction_parallel_spin.setEnabled(mode != "full_gpu")
        if is_full_gpu:
            idx = self.correction_device_combo.findData("gpu")
            if idx >= 0:
                self.correction_device_combo.setCurrentIndex(idx)
            if self.correction_parallel_spin.value() != 1:
                self.correction_parallel_spin.setValue(1)
        self.correction_device_combo.setEnabled(True)
    def _update_post_action_controls(self):
        action = self.post_action_combo.currentData()
        is_notify = action == "notify"
        self.post_notification_label.setVisible(is_notify)
        self.post_notification_input.setVisible(is_notify)

    def _create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(20)

        self.right_tabs = QTabWidget()
        self.right_tabs.setDocumentMode(True)

        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(12)
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setStyleSheet(
            f"""
            QTextEdit {{
                background-color: {AppTheme.PANELS};
                border: 1px solid {AppTheme.BORDER};
                border-radius: 12px;
                padding: 15px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 13px;
            }}
            """
        )
        log_layout.addWidget(self.log_widget)
        self.right_tabs.addTab(log_tab, "Журнал")

        post_tab = QWidget()
        post_layout = QVBoxLayout(post_tab)
        post_layout.setContentsMargins(0, 0, 0, 0)
        post_layout.setSpacing(12)
        post_form = QFormLayout()
        post_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.post_action_combo = QComboBox()
        self.post_action_combo.addItem("Ничего не делать", "none")
        self.post_action_combo.addItem("Открыть папку результатов", "open_folder")
        self.post_action_combo.addItem("Показать уведомление", "notify")
        self.post_action_combo.setStyleSheet(AppTheme.COMBOBOX_STYLE)
        post_form.addRow("Действие:", self.post_action_combo)
        self.post_notification_label = QLabel("Текст уведомления:")
        self.post_notification_input = QLineEdit()
        self.post_notification_input.setPlaceholderText("Обработка завершена")
        self.post_notification_input.setStyleSheet(
            f"background-color: {AppTheme.PANELS}; border: 1px solid {AppTheme.BORDER}; border-radius: 8px; padding: 8px 12px;"
        )
        post_form.addRow(self.post_notification_label, self.post_notification_input)
        post_layout.addLayout(post_form)
        post_layout.addStretch()
        self.right_tabs.addTab(post_tab, "Пост-обработка")
        self.post_action_combo.currentIndexChanged.connect(self._update_post_action_controls)
        self._update_post_action_controls()

        layout.addWidget(self.right_tabs)

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
        self.lmstudio_enabled_checkbox.setChecked(bool(self.config.get("lmstudio_enabled")))
        self.lmstudio_base_input.setText(self.config.get("lmstudio_base_url") or "")
        self.lmstudio_model_input.setText(self.config.get("lmstudio_model") or "")
        self.lmstudio_api_key_input.setText(self.config.get("lmstudio_api_key") or "")
        batch_val = int(
            self.config.get("lmstudio_batch_size")
            or self.config.get("correction_batch_size")
            or 40
        )
        self.lmstudio_batch_spin.setValue(batch_val)
        device = (self.config.get("device") or "cpu").lower()
        if device == "cuda":
            self.gpu_radio.setChecked(True)
        else:
            self.cpu_radio.setChecked(True)

        selected_formats = list(self.config.get("output_formats_multi") or ["srt"])
        if not selected_formats:
            selected_formats = ["srt"]
        for key, checkbox in self.output_format_checks.items():
            checkbox.setChecked(key in selected_formats)
        if not any(cb.isChecked() for cb in self.output_format_checks.values()):
            self.output_format_checks["srt"].setChecked(True)

        pipeline_mode = self.config.get("parallel_mode") or "balanced"
        idx = self.pipeline_mode_combo.findData(pipeline_mode)
        if idx >= 0:
            self.pipeline_mode_combo.setCurrentIndex(idx)
        else:
            self.pipeline_mode_combo.setCurrentIndex(0)

        self.transcription_parallel_spin.setValue(int(self.config.get("max_parallel_transcriptions") or 1))
        self.correction_parallel_spin.setValue(int(self.config.get("max_parallel_corrections") or 2))

        correction_device = self.config.get("correction_device") or "auto"
        idx = self.correction_device_combo.findData(correction_device)
        if idx >= 0:
            self.correction_device_combo.setCurrentIndex(idx)
        else:
            self.correction_device_combo.setCurrentIndex(0)

        batch_size = int(
            self.config.get("correction_batch_size")
            or self.config.get("llama_batch_size")
            or 40
        )
        self.correction_batch_spin.setValue(batch_size)

        action = self.config.get("post_processing_action") or "none"
        idx = self.post_action_combo.findData(action)
        if idx >= 0:
            self.post_action_combo.setCurrentIndex(idx)
        else:
            self.post_action_combo.setCurrentIndex(0)
        self.post_notification_input.setText(self.config.get("post_processing_notification_text") or "")
        self._update_post_action_controls()
        self._update_parallel_controls_state()

    def save_settings(self):
        self.config.set("output_dir", self.output_label.text())
        self.config.set("model_size", self.model_combo.currentText())
        self.config.set("language", self.lang_combo.currentText())
        self.config.set("translate_lang", self.translate_lang_combo.currentText())
        self.config.set("device", "cuda" if self.gpu_radio.isChecked() else "cpu")
        self.config.set("lmstudio_enabled", self.lmstudio_enabled_checkbox.isChecked())
        self.config.set("lmstudio_base_url", self.lmstudio_base_input.text().strip())
        self.config.set("lmstudio_model", self.lmstudio_model_input.text().strip())
        self.config.set("lmstudio_api_key", self.lmstudio_api_key_input.text().strip())

        selected_formats = [key for key, checkbox in self.output_format_checks.items() if checkbox.isChecked()]
        if not selected_formats:
            selected_formats = ["srt"]
        self.config.set("output_formats_multi", selected_formats)
        primary_format = selected_formats[0]
        if primary_format.startswith("txt"):
            self.config.set("output_format", "txt")
            self.config.set("include_txt_timestamps", primary_format in {"txt_ts", "txt_timestamps"})
        else:
            self.config.set("output_format", "srt")
            self.config.set("include_txt_timestamps", primary_format in {"txt_ts", "txt_timestamps"})

        self.config.set("parallel_mode", self.pipeline_mode_combo.currentData())
        self.config.set("max_parallel_transcriptions", self.transcription_parallel_spin.value())
        self.config.set("max_parallel_corrections", self.correction_parallel_spin.value())
        correction_device = self.correction_device_combo.currentData()
        self.config.set("correction_device", correction_device)
        self.config.set("correction_batch_size", self.correction_batch_spin.value())
        self.config.set("llama_batch_size", self.correction_batch_spin.value())
        self.config.set("lmstudio_batch_size", self.lmstudio_batch_spin.value())
        self.config.set("llama_gpu_layers", -1 if correction_device == "gpu" else 0)
        self.config.set("post_processing_action", self.post_action_combo.currentData())
        self.config.set("post_processing_notification_text", self.post_notification_input.text().strip())
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
            QMessageBox.warning(self, "Нет заданий", "Добавьте хотя бы один медиафайл для обработки.")
            return
        self.save_settings()
        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        correction_batch_size = int(
            self.config.get("correction_batch_size")
            or self.config.get("llama_batch_size")
            or 40
        )
        llama_batch_size = int(self.config.get("llama_batch_size") or correction_batch_size or 40)
        processing_settings = ProcessingSettings(
            pipeline_mode=self.config.get("parallel_mode") or "balanced",
            max_parallel_transcriptions=int(self.config.get("max_parallel_transcriptions") or 1),
            max_parallel_corrections=int(self.config.get("max_parallel_corrections") or 2),
            correction_device=self.config.get("correction_device") or "auto",
            correction_batch_size=correction_batch_size,
            release_whisper_after_batch=bool(self.config.get("release_whisper_after_batch")),
            llama_n_ctx=int(self.config.get("llama_n_ctx") or 4096),
            llama_batch_size=llama_batch_size,
            llama_gpu_layers=int(self.config.get("llama_gpu_layers") or 0),
            llama_main_gpu=int(self.config.get("llama_main_gpu") or 0),
            enable_cudnn_benchmark=bool(self.config.get("enable_cudnn_benchmark")),
            use_lmstudio=bool(self.config.get("lmstudio_enabled")),
            lmstudio_base_url=self.config.get("lmstudio_base_url") or "",
            lmstudio_model=self.config.get("lmstudio_model") or "",
            lmstudio_api_key=self.config.get("lmstudio_api_key") or "",
            lmstudio_batch_size=int(self.config.get("lmstudio_batch_size") or correction_batch_size),
        )
        self.worker.update_processing_settings(processing_settings)
        self.worker.resume_processing()
        self._post_action_triggered = False

        selected_formats = list(self.config.get("output_formats_multi") or ["srt"])
        if not selected_formats:
            selected_formats = ["srt"]

        for task_id, task in self.tasks.items():
            if task.status == "pending":
                task.output_dir = Path(self.config.get("output_dir"))
                task.output_format = "txt" if selected_formats[0].startswith("txt") else selected_formats[0]
                task.language = self.config.get("language")
                task.model_size = self.config.get("model_size")
                task.device = self.config.get("device")
                task.whisper_backend = self.config.get("whisper_backend") or "openai"
                task.faster_whisper_compute_type = self.config.get("faster_whisper_compute_type") or "int8"
                task.use_local_llm_correction = bool(self.config.get("use_local_llm_correction"))
                task.use_lmstudio = bool(self.config.get("lmstudio_enabled"))
                task.lmstudio_base_url = self.config.get("lmstudio_base_url") or ""
                task.lmstudio_model = self.config.get("lmstudio_model") or ""
                task.lmstudio_api_key = self.config.get("lmstudio_api_key") or ""
                task.lmstudio_batch_size = int(self.config.get("lmstudio_batch_size") or correction_batch_size)
                task.pipeline_mode = processing_settings.pipeline_mode
                task.max_parallel_transcriptions = processing_settings.max_parallel_transcriptions
                task.max_parallel_corrections = processing_settings.max_parallel_corrections
                task.correction_device = processing_settings.correction_device
                task.outputs = [
                    OutputRequest(format=fmt, include_timestamps=fmt in {"txt_ts", "txt_timestamps"})
                    for fmt in selected_formats
                ]
                task.include_timestamps = any(req.include_timestamps for req in task.outputs)
                task.error = None
                task.progress = 0
                task.status = "queued"
                task.result_paths = []
                if task_id in self.task_widgets:
                    self.task_widgets[task_id].update_progress(0)
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

    def on_task_completed(self, task_id, output_paths):
        paths: list[Path] = []
        if isinstance(output_paths, (list, tuple)):
            paths = [Path(p) for p in output_paths]
        elif isinstance(output_paths, str) and output_paths:
            paths = [Path(output_paths)]
        if task_id in self.tasks:
            self.tasks[task_id].status = "completed"
            self.tasks[task_id].result_paths = paths
        if task_id in self.task_widgets:
            self.task_widgets[task_id].show_translation_controls()
        self.check_all_tasks_done()

    def on_task_failed(self, task_id, error):
        if task_id in self.task_widgets:
            self.task_widgets[task_id].set_error(error)
        if task_id in self.tasks:
            self.tasks[task_id].status = "failed"
            self.tasks[task_id].error = error
            self.tasks[task_id].result_paths = []
        self.check_all_tasks_done()

    def handle_translation_request(self, task_id: str):
        task = self.tasks.get(task_id)
        if not task or not getattr(task, "result_paths", None):
            self.log_message("error", "Ошибка: файл ещё не готов к переводу.")
            return
        self.save_settings()
        target_lang = self.config.get("translate_lang")
        if task.language != "auto" and task.language == target_lang:
            self.log_message("warning", "Предупреждение: язык перевода совпадает с исходным.")
            QMessageBox.warning(self, "Предупреждение", "Язык перевода совпадает с исходным.")
            return
        widget = self.task_widgets.get(task_id)
        if widget:
            widget.set_status_translating()
        preferred = next((p for p in task.result_paths if p.suffix.lower() == ".srt"), task.result_paths[0])
        translation_task = TranslationTask(
            task_id=task_id,
            source_path=preferred,
            target_lang=target_lang,
            source_lang=self.config.get("language"),
            use_lmstudio=bool(self.config.get("use_local_llm_translation")) and bool(self.config.get("lmstudio_enabled")),
            lmstudio_base_url=self.config.get("lmstudio_base_url") or "",
            lmstudio_model=self.config.get("lmstudio_model") or "",
            lmstudio_api_key=self.config.get("lmstudio_api_key") or "",
            lmstudio_batch_size=int(self.config.get("lmstudio_batch_size") or self.config.get("correction_batch_size") or 40),
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
        if not self.tasks:
            self.process_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            return
        if any(t.status == "queued" for t in self.tasks.values()):
            return
        if all(t.status in ["completed", "failed", "pending"] for t in self.tasks.values()):
            self.process_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            if not self._post_action_triggered and any(
                t.status in ["completed", "failed"] for t in self.tasks.values()
            ):
                self._perform_post_processing_action()
                self._post_action_triggered = True


    def _perform_post_processing_action(self):
        action = (self.config.get("post_processing_action") or "none").lower()
        if action == "open_folder":
            output_dir = Path(self.config.get("output_dir"))
            if output_dir.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_dir)))
                self.log_message("info", f"Открываю папку: {output_dir}")
            else:
                self.log_message("warning", f"Папка не найдена: {output_dir}")
        elif action == "notify":
            message = self.config.get("post_processing_notification_text") or "Processing complete"
            QMessageBox.information(self, "Готово", message)

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
