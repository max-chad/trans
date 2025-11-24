from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set

from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.config import AppConfig
from app.lmstudio_client import LmStudioClient, ModelCapabilities, REASONING_DEFAULT_SELECTION

from .engine import TranscriptBatchProcessor, discover_transcripts
from .models import BatchReport, BatchRequest, ProcessingMode
from .provider_options import parse_provider_preferences

_REASONING_DEFAULT_LEVELS = ("high", "medium", "low")
_REASONING_DEFAULT_SELECTION_LABEL = "Enable reasoning (default config)"


class BatchWorker(QThread):
    progress = pyqtSignal(int, int)
    log_event = pyqtSignal(str, str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)
    paused = pyqtSignal()
    resumed = pyqtSignal()

    def __init__(self, request: BatchRequest):
        super().__init__()
        self.request = request
        self._state_lock = threading.Lock()
        self._pause_requested = False
        self._waiting_for_resume = False
        self._resume_event = threading.Event()
        self._resume_event.set()
        self._cancel_event = threading.Event()

    def request_pause(self) -> None:
        with self._state_lock:
            if self._waiting_for_resume:
                return
            self._pause_requested = True

    def resume_processing(self) -> None:
        with self._state_lock:
            self._pause_requested = False
            if self._waiting_for_resume:
                self._resume_event.set()

    def request_cancel(self) -> None:
        with self._state_lock:
            self._cancel_event.set()
            self._pause_requested = False
            self._resume_event.set()

    def run(self) -> None:
        started_at = time.perf_counter()
        processor = TranscriptBatchProcessor(self.request, log_callback=self._emit_log)
        report = BatchReport(self.request)
        try:
            had_progress = False
            cancelled = self._cancel_event.is_set()
            if not cancelled:
                for path, result, error, idx, total in processor.process_iter():
                    if self._cancel_event.is_set():
                        cancelled = True
                        break
                    had_progress = True
                    if error is None and result is not None:
                        report.record_success(result)
                    else:
                        reason = str(error) if error else "Unknown error"
                        report.record_failure(path, reason)
                    self.progress.emit(idx, total)
                    if self._cancel_event.is_set():
                        cancelled = True
                        break
                    self._wait_if_paused(idx < total)
            report.complete()
            json_path, md_path = processor.save_report(report)
            merged_story = processor.merge_story_outputs(report)
            payload = report.to_dict()
            payload["artifacts"] = {
                "json": str(json_path),
                "markdown": str(md_path),
                "merged_story": str(merged_story) if merged_story else None,
                "log": str(processor.log_path),
            }
            payload["had_progress"] = had_progress
            payload["cancelled"] = cancelled
            payload["elapsed_seconds"] = time.perf_counter() - started_at
            self.finished.emit(payload)
        except Exception as exc:  # pragma: no cover - UI thread handles feedback
            self.failed.emit(str(exc))
        finally:
            self._resume_event.set()

    def _wait_if_paused(self, has_more: bool) -> None:
        if self._cancel_event.is_set():
            return
        if not has_more:
            with self._state_lock:
                self._pause_requested = False
            return
        should_block = False
        with self._state_lock:
            if self._pause_requested and not self._waiting_for_resume:
                self._waiting_for_resume = True
                self._resume_event.clear()
                should_block = True
        if not should_block:
            return
        self.paused.emit()
        self._resume_event.wait()
        with self._state_lock:
            self._waiting_for_resume = False
        self.resumed.emit()

    def _emit_log(self, level: str, message: str) -> None:
        self.log_event.emit(level, message)


class TranscriptToolWindow(QWidget):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.worker: Optional[BatchWorker] = None
        self._awaiting_close_after_cancel = False
        self._batch_started_at: Optional[float] = None
        self._batch_started_wall: Optional[datetime] = None
        self._model_reasoning_supported = False
        self._capabilities_request_token = 0
        self._reasoning_values: list[Optional[str]] = [None]
        self.setWindowTitle("Transcript Batch Lab")
        self.resize(860, 640)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()

        self.input_edit = QLineEdit(str(self.config.get("input_dir") or Path.cwd()))
        browse_in_btn = QPushButton("Browse...")
        browse_in_btn.clicked.connect(self._select_input)
        form.addRow("Input directory", self._wrap_with_button(self.input_edit, browse_in_btn))

        default_output = Path(self.config.get("output_dir") or (Path.cwd() / "transcript_tool_output"))
        self.output_edit = QLineEdit(str(default_output))
        browse_out_btn = QPushButton("Browse...")
        browse_out_btn.clicked.connect(self._select_output)
        form.addRow("Output directory", self._wrap_with_button(self.output_edit, browse_out_btn))

        self.base_url_edit = QLineEdit(str(self.config.get("lmstudio_base_url") or "http://127.0.0.1:1234/v1"))
        form.addRow("LM Studio URL", self.base_url_edit)

        self.model_edit = QLineEdit(str(self.config.get("lmstudio_model") or ""))
        form.addRow("Model", self.model_edit)

        self.reasoning_combo = QComboBox()
        self.reasoning_combo.setEnabled(False)
        self._populate_reasoning_combo(_REASONING_DEFAULT_LEVELS, selected=self._initial_reasoning_selection())
        form.addRow("Reasoning effort", self.reasoning_combo)

        self.reasoning_status = QLabel("Enter LM Studio URL and model to refresh reasoning info.")
        self.reasoning_status.setWordWrap(True)
        form.addRow("", self.reasoning_status)

        self.provider_combo = QComboBox()
        self.provider_combo.setEditable(True)
        self.provider_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        for candidate in ["gmicloud/fp4", "openrouter/polaris-alpha"]:
            self.provider_combo.addItem(candidate)
        self.provider_combo.setEditText(self._initial_provider_selection())
        form.addRow("Preferred provider", self.provider_combo)

        self.api_key_edit = QLineEdit(str(self.config.get("lmstudio_api_key") or ""))
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("API key (optional)", self.api_key_edit)

        fetch_trigger = self._schedule_reasoning_fetch
        self.base_url_edit.editingFinished.connect(fetch_trigger)
        self.model_edit.editingFinished.connect(fetch_trigger)
        self.api_key_edit.editingFinished.connect(fetch_trigger)
        self._schedule_reasoning_fetch()

        self.lang_combo = QComboBox()
        self.lang_combo.setEditable(True)
        for lang in ["auto", "ru", "en", "es", "de", "fr"]:
            self.lang_combo.addItem(lang)
        default_lang = str(self.config.get("language") or "auto")
        idx = self.lang_combo.findText(default_lang)
        if idx >= 0:
            self.lang_combo.setCurrentIndex(idx)
        else:
            self.lang_combo.setEditText(default_lang)
        form.addRow("Language hint", self.lang_combo)

        modes_layout = QHBoxLayout()
        self.analysis_check = QCheckBox("Analysis")
        self.analysis_check.setChecked(self._mode_pref(ProcessingMode.ANALYSIS, True))
        self.rewrite_check = QCheckBox("Rewrite")
        self.rewrite_check.setChecked(self._mode_pref(ProcessingMode.REWRITE, False))
        self.deep_check = QCheckBox("Deep rewrite")
        self.deep_check.setChecked(self._mode_pref(ProcessingMode.DEEP_REWRITE, False))
        self.story_check = QCheckBox("Story")
        self.story_check.setChecked(self._mode_pref(ProcessingMode.STORY, False))
        modes_layout.addWidget(self.analysis_check)
        modes_layout.addWidget(self.rewrite_check)
        modes_layout.addWidget(self.deep_check)
        modes_layout.addWidget(self.story_check)
        modes_widget = QWidget()
        modes_widget.setLayout(modes_layout)
        form.addRow("Modes", modes_widget)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(0, 1000)
        self.batch_spin.setSpecialValueText("Auto (token limit)")
        stored_batch = self.config.get("lmstudio_batch_size")
        try:
            batch_value = int(stored_batch)
        except (TypeError, ValueError):
            batch_value = 0
        self.batch_spin.setValue(batch_value)
        form.addRow("Batch size (lines, 0 = auto)", self.batch_spin)

        self.prompt_spin = QSpinBox()
        self.prompt_spin.setRange(512, 500000)
        self.prompt_spin.setValue(int(self.config.get("lmstudio_prompt_token_limit") or 8192))
        form.addRow("Prompt limit", self.prompt_spin)


        packing_default = self.config.get("prompt_packing_enabled")
        if packing_default is None:
            packing_default = True
        self.packing_check = QCheckBox("Pack prompts across files when possible")
        self.packing_check.setChecked(bool(packing_default))
        form.addRow("Prompt packing", self.packing_check)

        resume_default = self.config.get("resume_enabled")
        if resume_default is None:
            resume_default = True
        self.resume_check = QCheckBox("Skip files that already have outputs")
        self.resume_check.setChecked(bool(resume_default))
        form.addRow("Auto resume", self.resume_check)

        layout.addLayout(form)

        controls = QHBoxLayout()
        self.scan_button = QPushButton("Scan")
        self.scan_button.clicked.connect(self._scan_files)
        controls.addWidget(self.scan_button)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self._start_batch)
        controls.addWidget(self.start_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self._pause_batch)
        controls.addWidget(self.pause_button)

        self.resume_button = QPushButton("Resume")
        self.resume_button.setEnabled(False)
        self.resume_button.clicked.connect(self._resume_batch)
        controls.addWidget(self.resume_button)

        controls.addStretch()

        self.status_label = QLabel("Idle")
        controls.addWidget(self.status_label)

        layout.addLayout(controls)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Batch log will appear here.")
        layout.addWidget(self.log_view, stretch=1)

    def _wrap_with_button(self, line_edit: QLineEdit, button: QPushButton) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit)
        layout.addWidget(button)
        return container

    def _select_input(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select input directory", self.input_edit.text())
        if directory:
            path = str(Path(directory).expanduser().resolve())
            self.input_edit.setText(path)
            self.config.set("input_dir", path)
            self.config.save()

    def _select_output(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select output directory", self.output_edit.text())
        if directory:
            path = str(Path(directory).expanduser().resolve())
            self.output_edit.setText(path)
            self.config.set("output_dir", path)
            self.config.save()

    def _scan_files(self) -> None:
        try:
            input_dir = Path(self.input_edit.text()).expanduser().resolve()
            output_dir = Path(self.output_edit.text()).expanduser().resolve()
            files = discover_transcripts(input_dir, ignore_output=output_dir)
        except Exception as exc:
            self._append_log("error", f"Scan failed: {exc}")
            self.status_label.setText("Scan failed")
            return
        self.status_label.setText(f"Found {len(files)} transcript(s)")
        self._append_log("info", f"Found {len(files)} transcript(s)")

    def _start_batch(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        try:
            request = self._build_request()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid input", str(exc))
            return
        self.progress.setValue(0)
        self.status_label.setText("Processing...")
        self.log_view.clear()
        self._batch_started_at = time.perf_counter()
        self._batch_started_wall = datetime.now()
        self._append_log("info", f"Batch started at {self._batch_started_wall.strftime('%H:%M:%S')}")
        self._set_controls_enabled(False)
        self.worker = BatchWorker(request)
        self.worker.progress.connect(self._on_progress)
        self.worker.log_event.connect(self._append_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.paused.connect(self._on_worker_paused)
        self.worker.resumed.connect(self._on_worker_resumed)
        self.pause_button.setEnabled(True)
        self.resume_button.setEnabled(False)
        self.worker.start()

    def _pause_batch(self) -> None:
        if not self.worker or not self.worker.isRunning():
            return
        self.worker.request_pause()
        self.pause_button.setEnabled(False)
        self.status_label.setText("Pausing after current file...")

    def _resume_batch(self) -> None:
        if not self.worker or not self.worker.isRunning():
            return
        self.worker.resume_processing()
        self.resume_button.setEnabled(False)
        self.status_label.setText("Processing...")

    def _build_request(self) -> BatchRequest:
        input_dir = Path(self.input_edit.text()).expanduser()
        output_dir = Path(self.output_edit.text()).expanduser()
        if not input_dir.exists():
            raise ValueError("Input directory does not exist.")
        operations = self._collect_modes()
        if not operations:
            raise ValueError("Select at least one mode.")
        base_url = self.base_url_edit.text().strip()
        model = self.model_edit.text().strip()
        needs_llm = any(
            mode
            in {
                ProcessingMode.ANALYSIS,
                ProcessingMode.REWRITE,
                ProcessingMode.DEEP_REWRITE,
                ProcessingMode.STORY,
            }
            for mode in operations
        )
        if needs_llm and (not base_url or not model):
            raise ValueError("LM Studio URL and model must be specified for the selected modes.")
        language_hint = self.lang_combo.currentText().strip() or "auto"
        merge_story_flag = bool(self.config.get("merge_story_output"))
        provider_preferences = self._resolve_provider_preferences()
        reasoning_effort = self._get_selected_reasoning_effort() if self._model_reasoning_supported else None

        request = BatchRequest(
            input_dir=input_dir.resolve(),
            output_dir=output_dir.resolve(),
            language_hint=language_hint,
            operations=operations,
            batch_size=self.batch_spin.value(),
            prompt_token_limit=self.prompt_spin.value(),
            base_url=base_url,
            model=model,
            api_key=self.api_key_edit.text().strip(),
            merge_story_output=merge_story_flag,
            provider_preferences=provider_preferences,
            reasoning_effort=reasoning_effort,
            resume=self.resume_check.isChecked(),
            prompt_packing=self.packing_check.isChecked(),
        )
        self._persist_config(request)
        return request

    def _collect_modes(self) -> Set[ProcessingMode]:
        modes: Set[ProcessingMode] = set()
        if self.analysis_check.isChecked():
            modes.add(ProcessingMode.ANALYSIS)
        if self.rewrite_check.isChecked():
            modes.add(ProcessingMode.REWRITE)
        if self.deep_check.isChecked():
            modes.add(ProcessingMode.DEEP_REWRITE)
        if self.story_check.isChecked():
            modes.add(ProcessingMode.STORY)
        return modes

    def _resolve_provider_preferences(self) -> Dict[str, Any]:
        provider_raw = self.config.get("lmstudio_provider_preferences")
        if provider_raw is None:
            provider_raw = self.config.get("provider_preferences")
        if provider_raw is not None:
            try:
                return parse_provider_preferences(provider_raw, allow_file_refs=True)
            except ValueError as exc:
                raise ValueError(f"Invalid provider options in config: {exc}") from exc
        provider_name = self.provider_combo.currentText().strip()
        if provider_name:
            return {"order": [provider_name]}
        return {}

    def _initial_provider_selection(self) -> str:
        override = self.config.get("lmstudio_provider_name")
        if override is None:
            override = self.config.get("provider_name")
        if override is not None:
            return str(override).strip()
        provider_raw = self.config.get("lmstudio_provider_preferences")
        if provider_raw is None:
            provider_raw = self.config.get("provider_preferences")
        if provider_raw is not None:
            try:
                preferences = parse_provider_preferences(provider_raw, allow_file_refs=True)
            except ValueError:
                return "gmicloud/fp4"
            order = preferences.get("order")
            if isinstance(order, (list, tuple)) and order:
                first = order[0]
                if isinstance(first, str):
                    trimmed = first.strip()
                    if trimmed:
                        return trimmed
            name = preferences.get("name")
            if isinstance(name, str):
                trimmed = name.strip()
                if trimmed:
                    return trimmed
                return ""
        return "gmicloud/fp4"

    def _populate_reasoning_combo(self, levels: Sequence[str], selected: Optional[str] = None) -> None:
        target = selected if selected is not None else self._get_selected_reasoning_effort()
        self.reasoning_combo.blockSignals(True)
        self.reasoning_combo.clear()
        self._reasoning_values = [None]
        self.reasoning_combo.addItem("Auto (model default)")
        self._reasoning_values.append(REASONING_DEFAULT_SELECTION)
        self.reasoning_combo.addItem(_REASONING_DEFAULT_SELECTION_LABEL)
        for level in levels:
            normalized = str(level).strip().lower()
            if not normalized:
                continue
            self.reasoning_combo.addItem(normalized.capitalize())
            self._reasoning_values.append(normalized)
        self._apply_reasoning_selection(target)
        self.reasoning_combo.blockSignals(False)

    def _apply_reasoning_selection(self, value: Optional[str]) -> None:
        normalized = value.strip().lower() if isinstance(value, str) else None
        for idx, stored in enumerate(self._reasoning_values):
            if stored == normalized:
                self.reasoning_combo.setCurrentIndex(idx)
                return
        self.reasoning_combo.setCurrentIndex(0)

    def _get_selected_reasoning_effort(self) -> Optional[str]:
        idx = self.reasoning_combo.currentIndex()
        if idx < 0 or idx >= len(self._reasoning_values):
            return None
        return self._reasoning_values[idx]

    def _initial_reasoning_selection(self) -> Optional[str]:
        stored = self.config.get("lmstudio_reasoning_effort")
        if stored is None:
            stored = self.config.get("reasoning_effort")
        if stored is None:
            return None
        normalized = str(stored).strip().lower()
        allowed = {*_REASONING_DEFAULT_LEVELS, REASONING_DEFAULT_SELECTION}
        return normalized if normalized in allowed else None

    def _schedule_reasoning_fetch(self) -> None:
        base_url = self.base_url_edit.text().strip()
        model = self.model_edit.text().strip()
        if not base_url or not model:
            self._model_reasoning_supported = False
            self._apply_reasoning_info(None, "Enter LM Studio URL and model to refresh reasoning info.", None)
            return
        self._capabilities_request_token += 1
        token = self._capabilities_request_token
        self.reasoning_status.setText("Refreshing reasoning info...")
        worker = threading.Thread(
            target=self._fetch_reasoning_worker,
            args=(base_url, model, self.api_key_edit.text().strip(), token),
            daemon=True,
        )
        worker.start()

    def _fetch_reasoning_worker(self, base_url: str, model: str, api_key: str, token: int) -> None:
        try:
            capabilities = LmStudioClient.fetch_model_capabilities(base_url, model=model, api_key=api_key)
            error = None
        except Exception as exc:
            capabilities = None
            error = str(exc)
        QTimer.singleShot(
            0,
            lambda caps=capabilities, err=error, tok=token: self._apply_reasoning_info(caps, err, tok),
        )

    def _apply_reasoning_info(
        self, capabilities: ModelCapabilities | None, error: Optional[str], token: Optional[int]
    ) -> None:
        if token is not None and token != self._capabilities_request_token:
            return
        previous = self._get_selected_reasoning_effort()
        if error:
            self._model_reasoning_supported = False
            self._populate_reasoning_combo(_REASONING_DEFAULT_LEVELS, selected=previous)
            self.reasoning_combo.setEnabled(False)
            self.reasoning_status.setText(f"Failed to fetch reasoning info: {error}")
            return
        if capabilities and capabilities.supports_reasoning:
            self._model_reasoning_supported = True
            levels = capabilities.reasoning_levels or _REASONING_DEFAULT_LEVELS
            self._populate_reasoning_combo(levels, selected=previous)
            self.reasoning_combo.setEnabled(True)
            self.reasoning_status.setText(
                "Model advertises reasoning effort levels: " + ", ".join(levels)
            )
        else:
            self._model_reasoning_supported = False
            self._populate_reasoning_combo(_REASONING_DEFAULT_LEVELS, selected=previous)
            self.reasoning_combo.setEnabled(False)
            self.reasoning_status.setText("Model does not advertise reasoning controls.")

    def _mode_pref(self, mode: ProcessingMode, default: bool) -> bool:
        value = self.config.get(f"mode_{mode.value}")
        if value is None:
            return default
        return bool(value)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in [
            self.input_edit,
            self.output_edit,
            self.base_url_edit,
            self.model_edit,
            self.reasoning_combo,
            self.provider_combo,
            self.api_key_edit,
            self.lang_combo,
            self.analysis_check,
            self.rewrite_check,
            self.deep_check,
            self.story_check,
            self.batch_spin,
            self.prompt_spin,
            self.packing_check,
            self.resume_check,
            self.scan_button,
            self.start_button,
        ]:
            widget.setEnabled(enabled)

    def _persist_config(self, request: BatchRequest | None = None) -> None:
        try:
            provider_name = self.provider_combo.currentText().strip()
            reasoning_effort = None
            if request is not None:
                language = request.language_hint
                base_url = request.base_url
                model = request.model
                api_key = request.api_key
                batch_size = request.batch_size
                prompt_limit = request.prompt_token_limit
                merge_story = request.merge_story_output
                operations = request.operations
                resume_enabled = request.resume
                packing_enabled = request.prompt_packing
                reasoning_effort = request.reasoning_effort
            else:
                language = self.lang_combo.currentText().strip() or "auto"
                base_url = self.base_url_edit.text().strip()
                model = self.model_edit.text().strip()
                api_key = self.api_key_edit.text().strip()
                batch_size = int(self.batch_spin.value())
                prompt_limit = int(self.prompt_spin.value())
                merge_story = bool(self.config.get("merge_story_output"))
                operations = self._collect_modes()
                resume_enabled = self.resume_check.isChecked()
                packing_enabled = self.packing_check.isChecked()
                reasoning_effort = self._get_selected_reasoning_effort()

            self.config.set("language", language)
            self.config.set("lmstudio_base_url", base_url)
            self.config.set("lmstudio_model", model)
            self.config.set("lmstudio_api_key", api_key)
            self.config.set("lmstudio_batch_size", batch_size)
            self.config.set("lmstudio_prompt_token_limit", prompt_limit)
            self.config.set("merge_story_output", merge_story)
            self.config.set("resume_enabled", bool(resume_enabled))
            self.config.set("prompt_packing_enabled", bool(packing_enabled))
            self.config.set("lmstudio_provider_name", provider_name or None)
            self.config.set("lmstudio_reasoning_effort", reasoning_effort)
            self._remember_modes(operations)
            self.config.save()
        except Exception as exc:  # pragma: no cover - best effort
            self._append_log("warning", f"Failed to save config: {exc}")

    def _remember_modes(self, operations: Set[ProcessingMode]) -> None:
        for mode in ProcessingMode:
            key = f"mode_{mode.value}"
            self.config.set(key, mode in operations)

    def _on_progress(self, current: int, total: int) -> None:
        if total <= 0:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            return
        self.progress.setRange(0, total)
        self.progress.setValue(current)
        self.status_label.setText(f"{current}/{total} processed")

    def _on_worker_paused(self) -> None:
        self.status_label.setText("Paused")
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(True)

    def _on_worker_resumed(self) -> None:
        self.status_label.setText("Processing...")
        self.pause_button.setEnabled(True)
        self.resume_button.setEnabled(False)

    def _on_finished(self, payload: dict) -> None:
        self._set_controls_enabled(True)
        cancelled = bool(payload.get("cancelled"))
        if not cancelled:
            self.progress.setValue(self.progress.maximum())
            self.status_label.setText("Completed")
        else:
            self.status_label.setText("Cancelled")
        log_level = "warning" if cancelled else "success"
        self._append_log(log_level, f"Report saved to {payload['artifacts']['json']}")
        summary = payload.get("summary", {})
        merged_story = payload.get("artifacts", {}).get("merged_story")
        log_path = payload.get("artifacts", {}).get("log")
        if merged_story:
            self._append_log("success", f"Merged story saved to {merged_story}")
        if log_path:
            self._append_log("info", f"Detailed log saved to {log_path}")
        if not self._awaiting_close_after_cancel:
            QMessageBox.information(
                self,
                "Batch cancelled" if cancelled else "Batch complete",
                self._build_completion_message(payload, summary, merged_story, log_path),
            )
        self._log_runtime(payload.get("elapsed_seconds"))
        self.worker = None
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)

    def _build_completion_message(
        self, payload: dict, summary: dict, merged_story: Optional[str], log_path: Optional[str]
    ) -> str:
        cancelled = bool(payload.get("cancelled"))
        lines = [
            f"Processed {summary.get('processed', 0)} transcript(s).",
            f"Report: {payload['artifacts']['json']}",
        ]
        if cancelled:
            lines.append("Processing stopped before all files were handled.")
        if merged_story:
            lines.append(f"Merged story: {merged_story}")
        if log_path:
            lines.append(f"Detailed log: {log_path}")
        return "\n".join(lines)

    def _on_failed(self, message: str) -> None:
        self._set_controls_enabled(True)
        self.status_label.setText("Failed")
        self._append_log("error", message)
        self._log_runtime(None)
        self.worker = None
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        QMessageBox.critical(self, "Batch failed", message)

    def _append_log(self, level: str, message: str) -> None:
        prefix = level.upper()
        self.log_view.appendPlainText(f"[{prefix}] {message}")

    def _log_runtime(self, elapsed_hint: Optional[float]) -> None:
        elapsed = elapsed_hint
        if elapsed is None and self._batch_started_at is not None:
            elapsed = time.perf_counter() - self._batch_started_at
        started_wall = self._batch_started_wall
        self._batch_started_at = None
        self._batch_started_wall = None
        if elapsed is None:
            return
        message = f"Batch runtime: {self._format_duration(elapsed)}"
        if started_wall:
            message += f" (started {started_wall.strftime('%H:%M:%S')})"
        self._append_log("info", message)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 1:
            return f"{seconds * 1000:.0f} ms"
        total_seconds = int(round(seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes:02d}m {secs:02d}s"
        if minutes:
            return f"{minutes}m {secs:02d}s"
        return f"{secs}s"

    def closeEvent(self, event: QCloseEvent) -> None:
        self._persist_config()
        if self.worker and self.worker.isRunning():
            if not self._awaiting_close_after_cancel:
                reply = QMessageBox.question(
                    self,
                    "Batch in progress",
                    "Processing is still running. Stop the batch and exit?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self._awaiting_close_after_cancel = True
                    self._append_log("warning", "Stopping batch before exiting...")
                    self.status_label.setText("Stopping...")
                    self.worker.request_cancel()
                    self.worker.finished.connect(self._close_after_worker)
                    self.worker.failed.connect(self._close_after_worker)
                event.ignore()
                return
            event.ignore()
            return
        super().closeEvent(event)

    def _close_after_worker(self, *args: object) -> None:
        if not self._awaiting_close_after_cancel:
            return
        self._awaiting_close_after_cancel = False
        self.close()


def launch_window(config: AppConfig) -> int:
    import signal
    import sys

    app = QApplication([sys.argv[0]])
    window = TranscriptToolWindow(config)
    window.show()

    interrupted = {"flag": False}

    def _handle_sigint(*_args: object) -> None:
        interrupted["flag"] = True

    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_sigint)

    def _process_interrupt() -> None:
        if not interrupted["flag"]:
            return
        interrupted["flag"] = False
        if window.worker and window.worker.isRunning():
            window._append_log("warning", "Interrupt received, stopping batch...")
            window.worker.request_cancel()
        app.quit()

    timer = QTimer()
    timer.setInterval(150)
    timer.timeout.connect(_process_interrupt)
    timer.start()

    try:
        return app.exec()
    finally:
        timer.stop()
        signal.signal(signal.SIGINT, previous_sigint)
