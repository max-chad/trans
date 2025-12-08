import sys
import os
import warnings
from pathlib import Path

from app.config import AppConfig, DEFAULT_CONFIG_PATH

# Suppress noisy TF32 deprecation warnings from torch on import.
warnings.filterwarnings(
    "ignore",
    message="Please use the new API settings to control TF32 behavior.*",
    module="torch.backends",
)
warnings.filterwarnings(
    "ignore",
    message="`torch.cuda.amp.custom_fwd\\(args\\.\\.\\.\\)` is deprecated.*",
    module="speechbrain.utils.autocast",
)


def launch_gui(config_path: Path | None = None) -> int:
    # Filter SYMLINK warning early
    warnings.filterwarnings("ignore", message=".*SYMLINK strategy.*", category=UserWarning)

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
    
    # Process events to ensure splash shows up
    app.processEvents()

    main_window = MainWindow(config)

    def show_main_window():
        splash.close()
        main_window.show()

    QTimer.singleShot(2500, show_main_window)

    return app.exec()


def main(argv: list[str] | None = None) -> int:
    try:
        # Construct absolute path to the .env file
        # This workaround forces HF/SpeechBrain to avoid symlinks on Windows
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        
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
    except BaseException as e:
        print(f"CRITICAL ERROR: {e!r}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Also catch sys.exit if possible, but sys.exit(main()) handles return code
    sys.exit(main())
   
