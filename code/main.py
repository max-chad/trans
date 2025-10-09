import sys
import os
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from ui.main_window import MainWindow
from ui.splash_screen import SplashScreen
from app.config import AppConfig


def main():
    app = QApplication(sys.argv)

    config_dir = Path.home() / ".video-transcriber"
    config = AppConfig(config_dir / "config.json")

    splash = SplashScreen()
    splash.show()

    main_window = MainWindow(config)

    def show_main_window():
        splash.close()
        main_window.show()

    QTimer.singleShot(2500, show_main_window)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()