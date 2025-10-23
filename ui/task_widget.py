from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

from app.models import TranscriptionTask
from .styles import AppTheme


class VideoTaskWidget(QFrame):
    remove_requested = pyqtSignal(str)
    translate_requested = pyqtSignal(str)

    def __init__(self, task: TranscriptionTask):
        super().__init__()
        self.task = task
        self.init_ui()

    def init_ui(self):
        self.setFrameShape(QFrame.Shape.StyledPanel)
        # Устанавливаем стили напрямую на QFrame, а не на дочерние виджеты
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {AppTheme.PANELS};
                border: 1px solid {AppTheme.BORDER};
                border-radius: 12px;
                padding: 12px;
            }}
        """)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 12, 15, 12)
        main_layout.setSpacing(15)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(5)

        self.name_label = QLabel(self.task.video_path.name)
        self.name_label.setStyleSheet(
            f"font-weight: bold; color: {AppTheme.TEXT_PRIMARY}; background: transparent; border: none;")
        info_layout.addWidget(self.name_label)

        self.status_label = QLabel("В ожидании")
        self.status_label.setStyleSheet(f"color: {AppTheme.TEXT_SECONDARY}; background: transparent; border: none;")
        info_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {AppTheme.BACKGROUND};
                border-radius: 3px;
                border: none;
            }}
            QProgressBar::chunk {{
                background-color: {AppTheme.ACCENT};
                border-radius: 3px;
            }}
        """)
        info_layout.addWidget(self.progress_bar)
        main_layout.addLayout(info_layout, 1)

        actions_layout = QVBoxLayout()

        self.translate_btn = QPushButton("Перевести")
        self.translate_btn.setStyleSheet(AppTheme.SECONDARY_BUTTON_STYLE)
        self.translate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.translate_btn.clicked.connect(self.request_translation)
        self.translate_btn.hide()
        actions_layout.addWidget(self.translate_btn)

        self.remove_btn = QPushButton("✕")
        self.remove_btn.setFixedSize(32, 32)
        self.remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.remove_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {AppTheme.TEXT_SECONDARY};
                border: none;
                font-size: 18px;
                font-weight: bold;
                border-radius: 16px;
            }}
            QPushButton:hover {{
                background-color: {AppTheme.BORDER};
                color: {AppTheme.ERROR};
            }}
        """)
        self.remove_btn.clicked.connect(lambda: self.remove_requested.emit(self.task.task_id))

        top_right_layout = QHBoxLayout()
        top_right_layout.addStretch()
        top_right_layout.addWidget(self.remove_btn)

        actions_wrapper_layout = QVBoxLayout()
        actions_wrapper_layout.addLayout(top_right_layout)
        actions_wrapper_layout.addStretch()
        actions_wrapper_layout.addWidget(self.translate_btn)
        actions_wrapper_layout.addStretch()

        main_layout.addLayout(actions_wrapper_layout)

    def request_translation(self):
        self.translate_requested.emit(self.task.task_id)

    def update_progress(self, value: int):
        self.progress_bar.setValue(value)
        if value < 100:
            self.status_label.setText(f"В работе... {value}%")
            self.status_label.setStyleSheet(f"color: {AppTheme.WARNING}; background: transparent; border: none;")
        else:
            self.status_label.setText("Завершено")
            self.status_label.setStyleSheet(f"color: {AppTheme.SUCCESS}; background: transparent; border: none;")

    def show_translation_controls(self):
        self.translate_btn.show()

    def set_status_translating(self):
        self.status_label.setText("Перевод...")
        self.status_label.setStyleSheet(f"color: {AppTheme.WARNING}; background: transparent; border: none;")
        self.translate_btn.setEnabled(False)

    def set_status_translation_complete(self):
        self.status_label.setText("Перевод завершен")
        self.status_label.setStyleSheet(f"color: {AppTheme.SUCCESS}; background: transparent; border: none;")
        self.translate_btn.setEnabled(True)

    def set_error(self, error: str):
        self.status_label.setText(f"Ошибка: {error[:50]}...")
        self.status_label.setToolTip(error)
        self.status_label.setStyleSheet(f"color: {AppTheme.ERROR}; background: transparent; border: none;")
        self.progress_bar.setStyleSheet(f"QProgressBar::chunk {{ background: {AppTheme.ERROR}; }}")
        self.translate_btn.setEnabled(True)
