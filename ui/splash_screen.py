from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

from .styles import AppTheme


class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.init_ui()

    def init_ui(self):
        self.setFixedSize(500, 300)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        container = QFrame()
        # Убраны стили border и border-radius для QFrame.
        # Вместо этого, контейнер будет иметь только фоновый цвет.
        container.setStyleSheet(f"""
            QFrame {{
                background-color: {AppTheme.PANELS};
                border-radius: 20px;
                /* Удалена бирюзовая рамка */
            }}
        """)

        container_layout = QVBoxLayout(container)
        container_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.setSpacing(20)

        logo_label = QLabel("🎬")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setStyleSheet("font-size: 80px; color: white; background: transparent;")
        container_layout.addWidget(logo_label)

        title_label = QLabel("Video Transcriber")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(
            f"color: {AppTheme.TEXT_PRIMARY}; font-size: 32px; font-weight: bold; background: transparent;")
        container_layout.addWidget(title_label)

        self.status_label = QLabel("Инициализация...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(f"color: {AppTheme.TEXT_SECONDARY}; font-size: 14px; background: transparent;")
        container_layout.addWidget(self.status_label)

        layout.addWidget(container)

        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)

    def set_status(self, message):
        self.status_label.setText(message)