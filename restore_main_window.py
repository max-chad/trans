from pathlib import Path

file_path = Path(r"c:\Users\max_chad\Music\transcriber\ui\main_window.py")
current_lines = file_path.read_text(encoding='utf-8').splitlines(keepends=True)

# We want lines starting from index 42 (line 43 in 1-based view)
tail_lines = current_lines[42:]

header_code = r'''from pathlib import Path
import time
import uuid

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import QDesktopServices

from app.config import AppConfig
from app.gpu_detection import GPUDetectionResult, detect_supported_nvidia_gpus
from app.lmstudio_client import validate_lmstudio_settings
from app.models import DeviceType, OutputRequest, ProcessingSettings, TranscriptionTask
from app.translator import TranslationWorker, TranslationTask
from app.worker import TranscriptionWorker
from .task_widget import VideoTaskWidget
from .styles import AppTheme


class MainWindow(QMainWindow):
    VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".mpg", ".mpeg", ".m4v", ".webm"}
    AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wma"}
    MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self._restore_maximized = False
        self._gpu_detection_cached_at = 0.0
        self.gpu_detection: GPUDetectionResult = detect_supported_nvidia_gpus()
        self._gpu_detection_cached_at = time.monotonic()
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
        self._apply_gpu_defaults()
        self._post_action_triggered = False

    def init_ui(self):
        self.setWindowTitle("Video Transcriber")
        width = int(self.config.get("window_width") or 1200)
        height = int(self.config.get("window_height") or 800)
        self.resize(width, height)
        self.setStyleSheet(AppTheme.GLOBAL_STYLE)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        left_panel = self._create_left_panel()
        right_panel = self._create_right_panel()
        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(right_panel)
        self.main_splitter.setStretchFactor(0, 4)
        self.main_splitter.setStretchFactor(1, 1)
        main_layout.addWidget(self.main_splitter)

    def _create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(20)
        self.drop_area = self._create_drop_area()
        self.drop_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.drop_area)

        self.left_splitter = QSplitter(Qt.Orientation.Vertical)
        self.left_splitter.setChildrenCollapsible(False)

        controls_wrapper = QWidget()
        controls_layout = QVBoxLayout(controls_wrapper)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(12)
        controls_panel = self._create_controls_panel()
        controls_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        controls_layout.addWidget(controls_panel)
        controls_layout.addStretch()
        self.left_splitter.addWidget(controls_wrapper)

        tasks_wrapper = QWidget()
        tasks_wrapper_layout = QVBoxLayout(tasks_wrapper)
        tasks_wrapper_layout.setContentsMargins(0, 0, 0, 0)
        tasks_wrapper_layout.setSpacing(12)
        tasks_header_layout = QHBoxLayout()
        tasks_header_label = QLabel("Очередь задач")
        tasks_header_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        tasks_header_layout.addWidget(tasks_header_label)
        tasks_header_layout.addStretch()
        clear_all_btn = QPushButton("Очистить все")
        clear_all_btn.setStyleSheet(AppTheme.SECONDARY_BUTTON_STYLE)
        clear_all_btn.clicked.connect(self.clear_all_tasks)
        tasks_header_layout.addWidget(clear_all_btn)
        tasks_wrapper_layout.addLayout(tasks_header_layout)

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
        tasks_wrapper_layout.addWidget(self.tasks_scroll)

        self.left_splitter.addWidget(tasks_wrapper)
        self.left_splitter.setStretchFactor(0, 2)
        self.left_splitter.setStretchFactor(1, 3)
        self.left_splitter.setSizes([360, 520])

        layout.addWidget(self.left_splitter, stretch=1)
        return panel

    def _get_gpu_detection(self, force_refresh: bool = False) -> GPUDetectionResult:
        cache_ttl = 60.0
        now = time.monotonic()
        if force_refresh or (now - self._gpu_detection_cached_at) > cache_ttl:
            self.gpu_detection = detect_supported_nvidia_gpus()
            self._gpu_detection_cached_at = time.monotonic()
        return self.gpu_detection

    def _apply_gpu_defaults(self, force_refresh: bool = False):
        detection = self._get_gpu_detection(force_refresh=force_refresh)
        torch_ready = detection.torch_usable
        gpu_present = detection.has_supported_series or detection.any_gpu
        names_text = ", ".join(detection.matched_names or detection.raw_names)

        if torch_ready:
            self.gpu_radio.setEnabled(True)
            self.hybrid_radio.setEnabled(True)
            preferred = (self.config.get("device") or "hybrid").lower()
            if preferred == "hybrid":
                self.hybrid_radio.setChecked(True)
            elif preferred == "cuda":
                self.gpu_radio.setChecked(True)
            else:
                self.cpu_radio.setChecked(True)
            if self.hybrid_radio.isChecked():
                self.config.set("device", "hybrid")
            elif self.gpu_radio.isChecked():
                self.config.set("device", "cuda")
            else:
                self.config.set("device", "cpu")
            if names_text:
                self.log_message("info", f"Обнаружен CUDA GPU: {names_text}")
            else:
                self.log_message("info", "Обнаружен CUDA GPU.")
        elif gpu_present:
            self.gpu_radio.setEnabled(False)
            self.hybrid_radio.setEnabled(False)
            self.cpu_radio.setChecked(True)
            self.config.set("device", "cpu")
            detail = names_text or "RTX устройство"
            self.log_message("warning", f"Обнаружен NVIDIA GPU ({detail}), но PyTorch сообщает об отсутствии CUDA. Использую CPU.")
        else:
            self.gpu_radio.setEnabled(False)
            self.hybrid_radio.setEnabled(False)
            self.cpu_radio.setChecked(True)
            self.config.set("device", "cpu")
            self.log_message("warning", "GPU не обнаружен — работа будет выполнена на CPU.")

    def _lmstudio_preflight(self, *, for_translation: bool = False) -> bool:
        if not bool(self.config.get("lmstudio_enabled")):
            return True
        base_url = self.config.get("lmstudio_base_url") or ""
        model = self.config.get("lmstudio_model") or ""
        api_key = self.config.get("lmstudio_api_key") or ""
        ok, reason = validate_lmstudio_settings(base_url, model, api_key)
        if ok:
            return True
        scope = "translation" if for_translation else "correction"
        message = f"{reason} {scope.capitalize()} will not start until the settings are fixed."
        self.log_message("error", message)
        QMessageBox.warning(self, "LM Studio configuration", message)
        return False

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

        # Row 0: Output Mode
        basic_layout.addWidget(QLabel("Режим сохранения:"), 0, 0)
        self.output_mode_combo = QComboBox()
        self.output_mode_combo.addItem("В папку с файлом", "source")
'''

full_content = header_code + "".join(tail_lines)
file_path.write_text(full_content, encoding='utf-8')
print(f"Restored file. New line count: {len(full_content.splitlines())}")
