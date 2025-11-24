import sys
from pathlib import Path

from app.config import AppConfig, DEFAULT_CONFIG_PATH


def launch_gui(config_path: Path | None = None) -> int:
    # Import torch before any PyQt modules to avoid Windows DLL init issues with CUDA wheels.
    import torch  # noqa: F401
    from ui.splash_screen import SplashScreen
    from ui.main_window import MainWindow
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer

    qt_args = [sys.argv[0]]
    app = QApplication(qt_args)

    target_config = config_path or DEFAULT_CONFIG_PATH
    config = AppConfig(target_config)

    splash = SplashScreen()
    splash.show()

    main_window = MainWindow(config)

    def show_main_window():
        splash.close()
        main_window.show()

    QTimer.singleShot(2500, show_main_window)

    return app.exec()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return launch_gui()
    if args == ["gui"]:
        return launch_gui()

    from app.cli import build_parser, run_cli

    parser = build_parser()
    parsed = parser.parse_args(args)
    if parsed.command in (None, "gui"):
        return launch_gui()
    return run_cli(parsed)


if __name__ == "__main__":
    sys.exit(main())
   
