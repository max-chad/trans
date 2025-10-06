import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from app.config import AppConfig
from ui.main_window import MainWindow
from ui.splash_screen import SplashScreen


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