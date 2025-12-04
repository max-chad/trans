from pathlib import Path
import time
import uuid

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import QDesktopServices

from app.config import AppConfig
from app.gpu_detection import GPUDetectionResult, detect_supported_nvidia_gpus
from app.lmstudio_client import validate_lmstudio_settings
from typing import Any
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
        # Pause processing until the user explicitly clicks "start".
        self.worker.stop_processing()
        self.translator = TranslationWorker()
        self.translator.translation_completed.connect(self.on_translation_completed)
        self.translator.translation_failed.connect(self.on_translation_failed)
        self.translator.log_message.connect(self.log_message)
        self.translator.start()
        self.init_ui()
        self.load_settings()
        self._apply_gpu_defaults()
        self._post_action_triggered = False
        self._processing_active = False

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
        self.output_mode_combo.addItem("В выбранную папку", "custom")
        self.output_mode_combo.setStyleSheet(AppTheme.COMBOBOX_STYLE)
        self.output_mode_combo.currentIndexChanged.connect(self._update_output_controls)
        basic_layout.addWidget(self.output_mode_combo, 0, 1, 1, 2)

        # Row 1: Output Folder
        self.output_folder_label_title = QLabel("Папка результата:")
        basic_layout.addWidget(self.output_folder_label_title, 1, 0)
        self.output_label = QLabel("")
        self.output_label.setStyleSheet(
            f"color: {AppTheme.TEXT_SECONDARY}; padding: 5px; border: 1px solid {AppTheme.BORDER}; border-radius: 5px;"
        )
        basic_layout.addWidget(self.output_label, 1, 1)
        self.output_btn = QPushButton("Выбрать")
        self.output_btn.setStyleSheet(AppTheme.SECONDARY_BUTTON_STYLE)
        self.output_btn.clicked.connect(self.select_output_dir)
        basic_layout.addWidget(self.output_btn, 1, 2)

        basic_layout.addWidget(QLabel("Язык распознавания:"), 2, 0)
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["auto", "ru", "en", "de", "fr", "es", "it", "uk", "pl"])
        self.lang_combo.setStyleSheet(AppTheme.COMBOBOX_STYLE)
        basic_layout.addWidget(self.lang_combo, 2, 1, 1, 2)

        basic_layout.addWidget(QLabel("Модель Whisper:"), 3, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large", "large-v3", "large-v3-turbo"])
        self.model_combo.setStyleSheet(AppTheme.COMBOBOX_STYLE)
        basic_layout.addWidget(self.model_combo, 3, 1, 1, 2)
        
        basic_layout.addWidget(QLabel("Точность вычислений:"), 4, 0)
        self.compute_type_combo = QComboBox()
        self.compute_type_combo.addItem("Auto", "auto")
        self.compute_type_combo.addItem("FP16 (Half)", "float16")
        self.compute_type_combo.addItem("FP32 (Full)", "float32")
        self.compute_type_combo.addItem("INT8_FP16 (Mixed)", "int8_float16")
        self.compute_type_combo.addItem("INT8 (Quantized)", "int8")
        self.compute_type_combo.addItem("INT8_BFLOAT16", "int8_bfloat16")
        self.compute_type_combo.addItem("BFLOAT16", "bfloat16")
        self.compute_type_combo.setStyleSheet(AppTheme.COMBOBOX_STYLE)
        basic_layout.addWidget(self.compute_type_combo, 4, 1, 1, 2)

        basic_layout.addWidget(QLabel("Устройство инференса:"), 5, 0)
        self.device_group = QButtonGroup()
        device_layout = QHBoxLayout()
        self.cpu_radio = QRadioButton("CPU")
        self.cpu_radio.setStyleSheet(AppTheme.RADIOBUTTON_STYLE)
        self.gpu_radio = QRadioButton("GPU (CUDA)")
        self.gpu_radio.setStyleSheet(AppTheme.RADIOBUTTON_STYLE)
        self.gpu_radio.setEnabled(self.gpu_detection.torch_usable)
        self.hybrid_radio = QRadioButton("Hybrid (CPU + GPU)")
        self.hybrid_radio.setStyleSheet(AppTheme.RADIOBUTTON_STYLE)
        self.hybrid_radio.setEnabled(self.gpu_detection.torch_usable)
        self.device_group.addButton(self.cpu_radio)
        self.device_group.addButton(self.gpu_radio)
        self.device_group.addButton(self.hybrid_radio)
        device_layout.addWidget(self.cpu_radio)
        device_layout.addWidget(self.gpu_radio)
        device_layout.addWidget(self.hybrid_radio)
        basic_layout.addLayout(device_layout, 5, 1, 1, 2)

        basic_layout.addWidget(QLabel("Форматы субтитров:"), 6, 0)
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
        basic_layout.addLayout(formats_layout, 6, 1, 1, 2)

        basic_layout.addWidget(QLabel("Язык перевода:"), 7, 0)
        self.translate_lang_combo = QComboBox()
        self.translate_lang_combo.addItems(["en", "ru", "de", "fr", "es", "it", "uk", "pl"])
        self.translate_lang_combo.setStyleSheet(AppTheme.COMBOBOX_STYLE)
        basic_layout.addWidget(self.translate_lang_combo, 7, 1, 1, 2)

        # Row 8: Processing Mode
        basic_layout.addWidget(QLabel("Режим обработки:"), 8, 0)
        self.pipeline_mode_combo = QComboBox()
        self.pipeline_mode_combo.addItem("Последовательно", "serial")
        self.pipeline_mode_combo.addItem("Пайплайн (GPU + CPU)", "balanced")
        self.pipeline_mode_combo.addItem("Transcribe then Correct (staged)", "staged")
        self.pipeline_mode_combo.addItem("Полностью на GPU (8 ГБ)", "full_gpu")
        self.pipeline_mode_combo.setStyleSheet(AppTheme.COMBOBOX_STYLE)
        basic_layout.addWidget(self.pipeline_mode_combo, 8, 1, 1, 2)

        # Row 9: Diarization Enable
        basic_layout.addWidget(QLabel("Diarization:"), 9, 0)
        self.diarization_enabled_checkbox = QCheckBox("Enable SpeechBrain (GPU/CPU)")
        self.diarization_enabled_checkbox.setStyleSheet(AppTheme.RADIOBUTTON_STYLE)
        basic_layout.addWidget(self.diarization_enabled_checkbox, 9, 1, 1, 2)

        # Row 10: Diarization Device
        basic_layout.addWidget(QLabel("Diarization device:"), 10, 0)
        self.diarization_device_combo = QComboBox()
        self.diarization_device_combo.addItem("Auto (prefer GPU)", "auto")
        self.diarization_device_combo.addItem("GPU (CUDA)", "cuda")
        self.diarization_device_combo.addItem("CPU", "cpu")
        self.diarization_device_combo.setStyleSheet(AppTheme.COMBOBOX_STYLE)
        basic_layout.addWidget(self.diarization_device_combo, 10, 1, 1, 2)

        # Row 11: Number of speakers
        basic_layout.addWidget(QLabel("Number of speakers (0 = auto):"), 11, 0)
        self.diarization_speakers_spin = QSpinBox()
        self.diarization_speakers_spin.setRange(0, 16)
        self.diarization_speakers_spin.setValue(0)
        self.diarization_speakers_spin.setStyleSheet(AppTheme.SPINBOX_STYLE)
        basic_layout.addWidget(self.diarization_speakers_spin, 11, 1)

        # Row 12: Diarization Compute Type
        basic_layout.addWidget(QLabel("Diarization precision:"), 12, 0)
        self.diarization_compute_type_combo = QComboBox()
        self.diarization_compute_type_combo.addItem("Auto", "auto")
        self.diarization_compute_type_combo.addItem("FP16 (Half)", "float16")
        self.diarization_compute_type_combo.addItem("FP32 (Full)", "float32")
        self.diarization_compute_type_combo.addItem("BFLOAT16", "bfloat16")
        self.diarization_compute_type_combo.setStyleSheet(AppTheme.COMBOBOX_STYLE)
        basic_layout.addWidget(self.diarization_compute_type_combo, 12, 1, 1, 2)

        basic_layout.setRowStretch(13, 1)
        self.settings_tabs.addTab(basic_tab, "Основные")

        advanced_tab = QWidget()
        advanced_layout = QGridLayout(advanced_tab)
        advanced_layout.setColumnStretch(1, 1)

        advanced_layout.addWidget(QLabel("LLM batch size:"), 0, 0)
        self.correction_batch_spin = QSpinBox()
        self.correction_batch_spin.setRange(4, 200)
        self.correction_batch_spin.setValue(40)
        self.correction_batch_spin.setStyleSheet(AppTheme.SPINBOX_STYLE)
        advanced_layout.addWidget(self.correction_batch_spin, 0, 1)
        
        advanced_layout.addWidget(QLabel("Batched Inference (Faster-Whisper):"), 1, 0)
        self.batched_inference_checkbox = QCheckBox("Enable Batched Inference")
        self.batched_inference_checkbox.setStyleSheet(AppTheme.RADIOBUTTON_STYLE)
        advanced_layout.addWidget(self.batched_inference_checkbox, 1, 1)
        
        advanced_layout.addWidget(QLabel("Inference Batch Size:"), 2, 0)
        self.batched_inference_batch_spin = QSpinBox()
        self.batched_inference_batch_spin.setRange(1, 128)
        self.batched_inference_batch_spin.setValue(16)
        self.batched_inference_batch_spin.setStyleSheet(AppTheme.SPINBOX_STYLE)
        advanced_layout.addWidget(self.batched_inference_batch_spin, 2, 1)

        self.deep_correction_checkbox = QCheckBox("Глубокая коррекция текста (замедляет обработку)")
        self.deep_correction_checkbox.setStyleSheet(AppTheme.RADIOBUTTON_STYLE)
        advanced_layout.addWidget(self.deep_correction_checkbox, 3, 0, 1, 4)

        advanced_layout.addWidget(QLabel("LM Studio enabled:"), 4, 0)
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

        advanced_layout.addWidget(QLabel("LM Studio prompt token limit:"), 9, 0)
        self.lmstudio_prompt_tokens_spin = QSpinBox()
        self.lmstudio_prompt_tokens_spin.setRange(512, 131072)
        self.lmstudio_prompt_tokens_spin.setSingleStep(512)
        self.lmstudio_prompt_tokens_spin.setValue(8192)
        self.lmstudio_prompt_tokens_spin.setStyleSheet(AppTheme.SPINBOX_STYLE)
        advanced_layout.addWidget(self.lmstudio_prompt_tokens_spin, 9, 1)
        advanced_layout.addWidget(QLabel("Таймаут загрузки LM Studio (сек):"), 10, 0)
        self.lmstudio_timeout_spin = QSpinBox()
        self.lmstudio_timeout_spin.setRange(60, 7200)
        self.lmstudio_timeout_spin.setSingleStep(30)
        self.lmstudio_timeout_spin.setValue(600)
        self.lmstudio_timeout_spin.setStyleSheet(AppTheme.SPINBOX_STYLE)
        advanced_layout.addWidget(self.lmstudio_timeout_spin, 10, 1)

        advanced_layout.addWidget(QLabel("Интервал опроса LM Studio (сек):"), 11, 0)
        self.lmstudio_poll_spin = QDoubleSpinBox()
        self.lmstudio_poll_spin.setRange(0.5, 15.0)
        self.lmstudio_poll_spin.setSingleStep(0.5)
        self.lmstudio_poll_spin.setValue(1.5)
        self.lmstudio_poll_spin.setDecimals(1)
        self.lmstudio_poll_spin.setStyleSheet(AppTheme.SPINBOX_STYLE)
        advanced_layout.addWidget(self.lmstudio_poll_spin, 11, 1)

        advanced_layout.setRowStretch(12, 1)
        self.settings_tabs.addTab(advanced_tab, "Дополнительно")

        return controls_group

    def _update_post_action_controls(self):
        action = self.post_action_combo.currentData()
        is_notify = action == "notify"
        self.post_notification_label.setVisible(is_notify)
        self.post_notification_input.setVisible(is_notify)

    def _update_output_controls(self):
        mode = self.output_mode_combo.currentData()
        is_custom = mode == "custom"
        self.output_folder_label_title.setVisible(is_custom)
        self.output_label.setVisible(is_custom)
        self.output_btn.setVisible(is_custom)

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
        width = int(self.config.get("window_width") or 1200)
        height = int(self.config.get("window_height") or 800)
        if not self.isMaximized():
            self.resize(width, height)
        splitter_sizes = self.config.get("splitter_sizes") or [int(width * 0.6), int(width * 0.4)]
        if isinstance(splitter_sizes, list) and len(splitter_sizes) == 2:
            try:
                self.main_splitter.setSizes([max(200, int(splitter_sizes[0])), max(200, int(splitter_sizes[1]))])
            except (TypeError, ValueError):
                self.main_splitter.setSizes([int(width * 0.6), int(width * 0.4)])
        left_sizes = self.config.get("left_splitter_sizes") or [int(height * 0.45), int(height * 0.55)]
        if hasattr(self, "left_splitter") and isinstance(left_sizes, list) and len(left_sizes) == 2:
            try:
                self.left_splitter.setSizes([max(200, int(left_sizes[0])), max(200, int(left_sizes[1]))])
            except (TypeError, ValueError):
                self.left_splitter.setSizes([360, 520])
        self._restore_maximized = bool(self.config.get("window_maximized"))
        
        output_mode = self.config.get("output_mode") or "source"
        idx = self.output_mode_combo.findData(output_mode)
        if idx >= 0:
            self.output_mode_combo.setCurrentIndex(idx)
        else:
            self.output_mode_combo.setCurrentIndex(0)
            
        self.output_label.setText(self.config.get("output_dir"))
        self._update_output_controls()
        self.model_combo.setCurrentText(self.config.get("model_size"))
        
        compute_type = self.config.get("faster_whisper_compute_type") or "float16"
        idx = self.compute_type_combo.findData(compute_type)
        if idx >= 0:
            self.compute_type_combo.setCurrentIndex(idx)
        else:
            self.compute_type_combo.setCurrentIndex(1) # Default to float16
            
        self.lang_combo.setCurrentText(self.config.get("language"))
        self.translate_lang_combo.setCurrentText(self.config.get("translate_lang") or "en")
        self.deep_correction_checkbox.setChecked(bool(self.config.get("deep_correction_enabled")))
        
        self.batched_inference_checkbox.setChecked(bool(self.config.get("batched_inference_enabled")))
        self.batched_inference_batch_spin.setValue(int(self.config.get("batched_inference_batch_size") or 16))
        
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
        prompt_limit = int(self.config.get("lmstudio_prompt_token_limit") or 8192)
        prompt_limit = max(self.lmstudio_prompt_tokens_spin.minimum(), min(prompt_limit, self.lmstudio_prompt_tokens_spin.maximum()))
        self.lmstudio_prompt_tokens_spin.setValue(prompt_limit)
        timeout_val = int(self.config.get("lmstudio_load_timeout") or 600)
        timeout_val = max(self.lmstudio_timeout_spin.minimum(), min(timeout_val, self.lmstudio_timeout_spin.maximum()))
        self.lmstudio_timeout_spin.setValue(timeout_val)
        poll_val = float(self.config.get("lmstudio_poll_interval") or 1.5)
        poll_val = max(self.lmstudio_poll_spin.minimum(), min(poll_val, self.lmstudio_poll_spin.maximum()))
        self.lmstudio_poll_spin.setValue(poll_val)
        device = (self.config.get("device") or "cpu").lower()
        if device == "hybrid" and self.hybrid_radio.isEnabled():
            self.hybrid_radio.setChecked(True)
        elif device == "cuda" and self.gpu_radio.isEnabled():
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

        self.diarization_enabled_checkbox.setChecked(bool(self.config.get("enable_diarization")))
        diar_device = (self.config.get("diarization_device") or "auto").lower()
        device_idx = self.diarization_device_combo.findData(diar_device)
        if device_idx >= 0:
            self.diarization_device_combo.setCurrentIndex(device_idx)
        num_speakers = int(self.config.get("diarization_num_speakers") or 0)
        num_speakers = max(self.diarization_speakers_spin.minimum(), min(num_speakers, self.diarization_speakers_spin.maximum()))
        self.diarization_speakers_spin.setValue(num_speakers)
        try:
            self.worker.set_diarization_device(diar_device)
        except Exception:
            pass
            
        diar_compute = self.config.get("diarization_compute_type") or "auto"
        idx = self.diarization_compute_type_combo.findData(diar_compute)
        if idx >= 0:
            self.diarization_compute_type_combo.setCurrentIndex(idx)
        else:
            self.diarization_compute_type_combo.setCurrentIndex(0)

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
        if self.config.load_warning:
            self.log_message("warning", self.config.load_warning)
            QMessageBox.warning(self, "Config warning", self.config.load_warning)

        # Connect signals for immediate saving
        self.output_mode_combo.currentIndexChanged.connect(lambda: self.on_setting_changed("output_mode", self.output_mode_combo.currentData()))
        self.lang_combo.currentTextChanged.connect(lambda t: self.on_setting_changed("language", t))
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        self.compute_type_combo.currentIndexChanged.connect(lambda: self.on_setting_changed("faster_whisper_compute_type", self.compute_type_combo.currentData()))
        self.translate_lang_combo.currentTextChanged.connect(lambda t: self.on_setting_changed("translate_lang", t))
        self.pipeline_mode_combo.currentIndexChanged.connect(lambda: self.on_setting_changed("parallel_mode", self.pipeline_mode_combo.currentData()))
        self.diarization_device_combo.currentIndexChanged.connect(lambda: self.on_setting_changed("diarization_device", self.diarization_device_combo.currentData()))
        self.diarization_compute_type_combo.currentIndexChanged.connect(lambda: self.on_setting_changed("diarization_compute_type", self.diarization_compute_type_combo.currentData()))
        self.post_action_combo.currentIndexChanged.connect(lambda: self.on_setting_changed("post_processing_action", self.post_action_combo.currentData()))
        
        self.device_group.buttonClicked.connect(self.on_device_changed)
        
        self.deep_correction_checkbox.toggled.connect(lambda c: self.on_setting_changed("deep_correction_enabled", c))
        self.batched_inference_checkbox.toggled.connect(lambda c: self.on_setting_changed("batched_inference_enabled", c))
        self.lmstudio_enabled_checkbox.toggled.connect(lambda c: self.on_setting_changed("lmstudio_enabled", c))
        self.diarization_enabled_checkbox.toggled.connect(lambda c: self.on_setting_changed("enable_diarization", c))
        
        self.correction_batch_spin.valueChanged.connect(lambda v: self.on_setting_changed("correction_batch_size", v))
        self.batched_inference_batch_spin.valueChanged.connect(lambda v: self.on_setting_changed("batched_inference_batch_size", v))
        self.lmstudio_batch_spin.valueChanged.connect(lambda v: self.on_setting_changed("lmstudio_batch_size", v))
        self.lmstudio_prompt_tokens_spin.valueChanged.connect(lambda v: self.on_setting_changed("lmstudio_prompt_token_limit", v))
        self.lmstudio_timeout_spin.valueChanged.connect(lambda v: self.on_setting_changed("lmstudio_load_timeout", v))
        self.lmstudio_poll_spin.valueChanged.connect(lambda v: self.on_setting_changed("lmstudio_poll_interval", v))
        self.diarization_speakers_spin.valueChanged.connect(lambda v: self.on_setting_changed("diarization_num_speakers", v))
        
        self.lmstudio_base_input.editingFinished.connect(lambda: self.on_setting_changed("lmstudio_base_url", self.lmstudio_base_input.text().strip()))
        self.lmstudio_model_input.editingFinished.connect(lambda: self.on_setting_changed("lmstudio_model", self.lmstudio_model_input.text().strip()))
        self.lmstudio_api_key_input.editingFinished.connect(lambda: self.on_setting_changed("lmstudio_api_key", self.lmstudio_api_key_input.text().strip()))
        self.post_notification_input.editingFinished.connect(lambda: self.on_setting_changed("post_processing_notification_text", self.post_notification_input.text().strip()))
        
        for key, checkbox in self.output_format_checks.items():
            checkbox.toggled.connect(self.on_output_format_changed)

    def on_setting_changed(self, key: str, value: Any):
        self.config.set(key, value)

    def on_model_changed(self, text: str):
        self.config.set("model_size", text)
        self.log_message("info", f"Model changed to: {text}")

    def on_device_changed(self, button):
        if button == self.hybrid_radio:
            self.config.set("device", "hybrid")
        elif button == self.gpu_radio:
            self.config.set("device", "cuda")
        else:
            self.config.set("device", "cpu")

    def on_output_format_changed(self):
        selected_formats = [key for key, checkbox in self.output_format_checks.items() if checkbox.isChecked()]
        if not selected_formats:
            # Prevent unchecking all
            sender = self.sender()
            if sender:
                sender.blockSignals(True)
                sender.setChecked(True)
                sender.blockSignals(False)
            return
            
        self.config.set("output_formats_multi", selected_formats)
        primary_format = selected_formats[0]
        if primary_format.startswith("txt"):
            self.config.set("output_format", "txt")
            self.config.set("include_txt_timestamps", primary_format in {"txt_ts", "txt_timestamps"})
        else:
            self.config.set("output_format", "srt")
            self.config.set("include_txt_timestamps", primary_format in {"txt_ts", "txt_timestamps"})

    def save_settings(self):
        if self.isMaximized():
            rect = self.normalGeometry()
            width, height = rect.width(), rect.height()
            maximized = True
        else:
            size = self.size()
            width, height = size.width(), size.height()
            maximized = False
        self.config.set("window_width", int(width))
        self.config.set("window_height", int(height))
        self.config.set("window_maximized", maximized)
        self.config.set("splitter_sizes", self.main_splitter.sizes())
        if hasattr(self, "left_splitter"):
            self.config.set("left_splitter_sizes", self.left_splitter.sizes())
        self.config.set("output_mode", self.output_mode_combo.currentData())
        self.config.set("output_dir", self.output_label.text())
        self.config.set("output_dir", self.output_label.text())
        self.config.set("model_size", self.model_combo.currentText())
        self.config.set("faster_whisper_compute_type", self.compute_type_combo.currentData())
        self.config.set("language", self.lang_combo.currentText())
        self.config.set("translate_lang", self.translate_lang_combo.currentText())
        if self.hybrid_radio.isChecked():
            self.config.set("device", "hybrid")
        elif self.gpu_radio.isChecked():
            self.config.set("device", "cuda")
        else:
            self.config.set("device", "cpu")
        self.config.set("deep_correction_enabled", self.deep_correction_checkbox.isChecked())
        self.config.set("batched_inference_enabled", self.batched_inference_checkbox.isChecked())
        self.config.set("batched_inference_batch_size", self.batched_inference_batch_spin.value())
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
        self.config.set("enable_diarization", self.diarization_enabled_checkbox.isChecked())
        self.config.set("diarization_device", self.diarization_device_combo.currentData())
        self.config.set("diarization_compute_type", self.diarization_compute_type_combo.currentData())
        self.config.set("diarization_num_speakers", int(self.diarization_speakers_spin.value()))
        self.config.set("correction_batch_size", self.correction_batch_spin.value())
        self.config.set("llama_batch_size", self.correction_batch_spin.value())
        self.config.set("lmstudio_batch_size", self.lmstudio_batch_spin.value())
        self.config.set("lmstudio_prompt_token_limit", self.lmstudio_prompt_tokens_spin.value())
        self.config.set("lmstudio_load_timeout", self.lmstudio_timeout_spin.value())
        self.config.set("lmstudio_poll_interval", float(self.lmstudio_poll_spin.value()))
        self.config.set("post_processing_action", self.post_action_combo.currentData())
        self.config.set("post_processing_notification_text", self.post_notification_input.text().strip())
        try:
            self.worker.set_diarization_device(self.diarization_device_combo.currentData())
        except Exception:
            pass
    def browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите видео или аудио файлы",
            "",
            "Медиа файлы (*.mp4 *.mkv *.avi *.mov *.mpg *.mpeg *.m4v *.webm *.wav *.mp3 *.flac *.m4a *.aac *.ogg *.oga *.opus *.wma)"
        )
        if files:
            self.add_video_files([Path(f) for f in files])

    def select_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения результатов")
        if directory:
            self.output_label.setText(directory)

    def _is_supported_media(self, path: Path) -> bool:
        return path.suffix.lower() in self.MEDIA_EXTENSIONS

    def add_videos_from_directory(self, directory: Path):
        resolved_dir = directory.resolve()
        if not resolved_dir.exists() or not resolved_dir.is_dir():
            self.log_message("warning", f"Папка недоступна: {directory}")
            return
        video_files = [
            path for path in resolved_dir.rglob("*")
            if path.is_file() and self._is_supported_media(path)
        ]
        if not video_files:
            self.log_message("warning", f"Видео не найдено в папке: {resolved_dir}")
            return
        self.log_message("info", f"Добавляю {len(video_files)} файлов из {resolved_dir}")
        self.add_video_files(video_files, source_root=resolved_dir, use_video_dir_as_output=True)

    def add_video_files(self, file_paths: list[Path], *, source_root: Path | None = None, use_video_dir_as_output: bool = False):
        root = source_root.resolve() if source_root else None
        
        # Pre-fetch settings
        backend = self.config.get("whisper_backend") or "faster"
        if backend not in {"faster", "openai"}:
            backend = "faster"
        if backend not in {"faster", "openai"}:
            backend = "faster"
        compute_type = self.config.get("faster_whisper_compute_type") or "float16"
        batched_enabled = bool(self.config.get("batched_inference_enabled"))
        batched_size = int(self.config.get("batched_inference_batch_size") or 16)
        
        use_correction = bool(self.config.get("use_local_llm_correction"))
        deep_correction = bool(self.config.get("deep_correction_enabled"))
        use_lmstudio = bool(self.config.get("lmstudio_enabled"))
        lm_base = self.config.get("lmstudio_base_url") or ""
        lm_model = self.config.get("lmstudio_model") or ""
        lm_key = self.config.get("lmstudio_api_key") or ""
        
        batch_size = int(self.config.get("correction_batch_size") or 40)
        lm_batch = int(self.config.get("lmstudio_batch_size") or batch_size)
        prompt_limit = int(self.config.get("lmstudio_prompt_token_limit") or 8192)
        load_timeout = int(self.config.get("lmstudio_load_timeout") or 600)
        poll_interval = float(self.config.get("lmstudio_poll_interval") or 1.5)
        pipeline_mode = self.config.get("parallel_mode") or "balanced"
        enable_diarization = bool(self.config.get("enable_diarization"))
        diarization_num_speakers = int(self.config.get("diarization_num_speakers") or 0)
        diarization_device = (self.config.get("diarization_device") or "auto")
        device_pref = (self.config.get("device") or "cpu").lower()
        if device_pref not in {"cpu", "cuda", "hybrid", "auto"}:
            device_pref = "cpu"
        
        selected_formats = list(self.config.get("output_formats_multi") or ["srt"])
        if not selected_formats:
            selected_formats = ["srt"]

        try:
            self.worker.set_diarization_device(diarization_device)
        except Exception:
            pass

        for path in file_paths:
            if path.is_dir():
                self.add_videos_from_directory(path)
                continue
            if not path.exists():
                self.log_message("warning", f"Файл не найден: {path}")
                continue
            if not self._is_supported_media(path):
                continue
            resolved_path = path.resolve()
            if any(task.video_path.resolve() == resolved_path for task in self.tasks.values()):
                self.log_message("warning", f"Видео '{path.name}' уже добавлено и не будет продублировано.")
                continue

            task_id = f"task_{uuid.uuid4().hex}"
            out_dir = Path(self.config.get("output_dir"))
            if use_video_dir_as_output:
                out_dir = resolved_path.parent
            elif self.config.get("output_mode") == "source":
                out_dir = resolved_path.parent
                
            task = TranscriptionTask(
                task_id=task_id,
                video_path=resolved_path,
                output_dir=out_dir,
                output_format=self.config.get("output_format") or "srt",
                language=self.config.get("language") or "auto",
                model_size=self.config.get("model_size") or "base",
                source_root=root
            )
            
            task.device = device_pref
            task.whisper_backend = backend
            task.faster_whisper_compute_type = compute_type
            task.batched_inference_enabled = batched_enabled
            task.batched_inference_batch_size = batched_size
            task.use_local_llm_correction = use_correction
            task.deep_correction = deep_correction
            task.use_lmstudio = use_lmstudio
            task.lmstudio_base_url = lm_base
            task.lmstudio_model = lm_model
            task.lmstudio_api_key = lm_key
            task.lmstudio_batch_size = lm_batch
            task.lmstudio_prompt_token_limit = prompt_limit
            task.lmstudio_load_timeout = load_timeout
            task.lmstudio_poll_interval = poll_interval
            task.pipeline_mode = pipeline_mode
            task.enable_diarization = enable_diarization
            task.num_speakers = diarization_num_speakers if diarization_num_speakers > 0 else None
            task.diarization_device = diarization_device
            task.diarization_compute_type = self.config.get("diarization_compute_type") or "auto"
            
            task.outputs = [
                OutputRequest(format=fmt, include_timestamps=fmt in {"txt_ts", "txt_timestamps"})
                for fmt in selected_formats
            ]
            task.include_timestamps = any(req.include_timestamps for req in task.outputs)
            task.error = None
            task.progress = 0
            task.status = "pending"
            task.result_paths = []
            
            self.tasks[task_id] = task
            
            # Create widget
            widget = VideoTaskWidget(task)
            widget.remove_requested.connect(self.remove_task)
            widget.translate_requested.connect(self.handle_translation_request)
            self.task_widgets[task_id] = widget
            self.tasks_layout.insertWidget(self.tasks_layout.count() - 1, widget)
            if self._processing_active:
                task.status = "queued"
                self.worker.add_task(task)
            
        self.log_message("info", f"Добавлено в очередь {len(self.tasks)} задач.")
        self.check_all_tasks_done()

    def remove_task(self, task_id: str):
        if task_id in self.tasks:
            del self.tasks[task_id]
        if task_id in self.task_widgets:
            widget = self.task_widgets.pop(task_id)
            self.tasks_layout.removeWidget(widget)
            widget.deleteLater()
        self.check_all_tasks_done()

    def clear_all_tasks(self):
        if any(t.status == "processing" for t in self.tasks.values()):
             QMessageBox.warning(self, "Ошибка", "Нельзя очистить список во время обработки.")
             return
        
        ids = list(self.tasks.keys())
        for tid in ids:
            self.remove_task(tid)
        self.log_message("info", "Список задач очищен.")

    def _build_processing_settings(self) -> ProcessingSettings:
        return ProcessingSettings(
            pipeline_mode=self.config.get("parallel_mode"),
            correction_gpu_layers=0,
            correction_batch_size=int(self.config.get("correction_batch_size") or 40),
            release_whisper_after_batch=bool(self.config.get("release_whisper_after_batch")),
            deep_correction=bool(self.config.get("deep_correction_enabled")),
            llama_n_ctx=int(self.config.get("llama_n_ctx") or 4096),
            llama_batch_size=int(self.config.get("llama_batch_size") or 40),
            llama_gpu_layers=int(self.config.get("llama_gpu_layers") or 0),
            llama_main_gpu=int(self.config.get("llama_main_gpu") or 0),
            enable_cudnn_benchmark=bool(self.config.get("enable_cudnn_benchmark")),
            use_lmstudio=bool(self.config.get("lmstudio_enabled")),
            lmstudio_base_url=self.config.get("lmstudio_base_url") or "",
            lmstudio_model=self.config.get("lmstudio_model") or "",
            lmstudio_api_key=self.config.get("lmstudio_api_key") or "",
            lmstudio_batch_size=int(self.config.get("lmstudio_batch_size") or 40),
            lmstudio_prompt_token_limit=int(self.config.get("lmstudio_prompt_token_limit") or 8192),
            lmstudio_load_timeout=float(self.config.get("lmstudio_load_timeout") or 600.0),
            lmstudio_poll_interval=float(self.config.get("lmstudio_poll_interval") or 1.5),
            diarization_device=self.config.get("diarization_device") or "auto",
            diarization_threshold=float(self.config.get("diarization_threshold") or 0.8),
            diarization_compute_type=self.config.get("diarization_compute_type") or "auto",
        )

    def start_processing(self):
        if not self.tasks:
            QMessageBox.warning(self, "Ошибка", "Нет задач для обработки.")
            return
            
        queued = [t for t in self.tasks.values() if t.status in ("queued", "pending", "failed")]
        if not queued:
             QMessageBox.information(self, "Info", "Все задачи уже выполнены.")
             return

        self._processing_active = True
        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker.update_processing_settings(self._build_processing_settings())
        self.worker.resume_processing()
        
        count = 0
        for task in queued:
            # Reset status if needed
            if task.status in ("pending", "failed"):
                task.status = "queued"
                task.error = None
                task.progress = 0
                if task.task_id in self.task_widgets:
                    self.task_widgets[task.task_id].update_progress(0)
                    self.task_widgets[task.task_id].set_error("")
            
            # Add to worker queue (since queue is cleared on stop)
            self.worker.add_task(task)
            count += 1
            
        self.log_message("info", f"Начало обработки... ({count} задач)")
    def stop_processing(self):
        self.log_message("warning", "Обработка всех задач остановлена.")
        self._processing_active = False
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
        wants_lmstudio = bool(self.config.get("use_local_llm_translation")) and bool(self.config.get("lmstudio_enabled"))
        if not wants_lmstudio:
            self.log_message("warning", "Translation requires LM Studio. Enable it in settings.")
            if widget:
                widget.set_error("LM Studio is required for translation.")
            return
        if not self._lmstudio_preflight(for_translation=True):
            if widget:
                widget.set_error("Translation cancelled: fix LM Studio settings.")
            return
        if widget:
            widget.set_status_translating()
        prompt_limit = int(self.config.get("lmstudio_prompt_token_limit") or 8192)
        load_timeout = int(self.config.get("lmstudio_load_timeout") or 600)
        poll_interval = float(self.config.get("lmstudio_poll_interval") or 1.5)
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
            lmstudio_prompt_token_limit=prompt_limit,
            lmstudio_load_timeout=load_timeout,
            lmstudio_poll_interval=poll_interval,
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
            self._processing_active = False
            return
        if any(t.status == "queued" for t in self.tasks.values()):
            return
        if all(t.status in ["completed", "failed", "pending"] for t in self.tasks.values()):
            self.process_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self._processing_active = False
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

    def showEvent(self, event):
        super().showEvent(event)
        if self._restore_maximized:
            self.showMaximized()
            self._restore_maximized = False

    def closeEvent(self, event):
        self.save_settings()
        self.worker.stop()
        self.translator.stop()
        self.worker.wait()
        self.translator.wait()
        event.accept()
