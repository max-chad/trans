import argparse
import sys
import threading
from pathlib import Path
from typing import Callable, List, Optional

from .config import AppConfig
from .models import OutputRequest, ProcessingSettings, TranscriptionTask
from .worker import TranscriptionWorker

DEFAULT_CONFIG_PATH = Path.home() / ".video-transcriber" / "config.json"


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI entry point with subcommands and shared options."""
    parser = argparse.ArgumentParser(
        prog="video-transcriber",
        description="Инструмент транскрибации с поддержкой GUI и консольного режима.",
    )
    subparsers = parser.add_subparsers(dest="command")

    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Запустить транскрибацию файла через консоль.",
    )
    transcribe_parser.add_argument(
        "input",
        type=Path,
        help="Путь к аудио или видео файлу.",
    )
    transcribe_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Каталог, куда будут сохранены результаты (по умолчанию из конфигурации).",
    )
    transcribe_parser.add_argument(
        "-f",
        "--format",
        action="append",
        dest="formats",
        help="Формат результата (можно указать несколько: srt, txt, txt_ts, vtt и т.д.).",
    )
    transcribe_parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Добавлять временные метки в текстовые выходы (txt).",
    )
    transcribe_parser.add_argument(
        "-l",
        "--language",
        help="Язык распознавания (например, en, ru, auto).",
    )
    transcribe_parser.add_argument(
        "-m",
        "--model",
        help="Размер модели Whisper (tiny, base, small, medium, large и т.д.).",
    )
    transcribe_parser.add_argument(
        "-d",
        "--device",
        choices=["cpu", "cuda", "hybrid", "auto"],
        help="Устройство для инференса (cpu, cuda, hybrid или auto).",
    )
    transcribe_parser.add_argument(
        "--no-correction",
        action="store_true",
        help="Отключить этап локальной коррекции текста.",
    )
    transcribe_parser.add_argument(
        "--lmstudio-url",
        help="URL сервера LM Studio (переопределяет значение из конфигурации).",
    )
    transcribe_parser.add_argument(
        "--lmstudio-model",
        help="Имя модели LM Studio (переопределяет значение из конфигурации).",
    )
    transcribe_parser.add_argument(
        "--lmstudio-api-key",
        help="API ключ для LM Studio (если требуется).",
    )
    transcribe_parser.add_argument(
        "--config",
        type=Path,
        help="Пользовательский путь к конфигурационному файлу.",
    )
    transcribe_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Отключить вывод прогресса и логов.",
    )
    transcribe_parser.set_defaults(command="transcribe")

    gui_parser = subparsers.add_parser(
        "gui",
        help="Запустить графический интерфейс (аналог поведения по умолчанию).",
    )
    gui_parser.set_defaults(command="gui")

    return parser


def run_cli(args: argparse.Namespace) -> int:
    # The CLI currently exposes only the transcription workflow, so reject unknown commands early.
    if args.command != "transcribe":
        print("Неизвестная команда. Используйте 'gui' или 'transcribe'.", file=sys.stderr)
        return 2
    return _run_transcribe_command(args)


def _run_transcribe_command(args: argparse.Namespace) -> int:
    # Bridge worker callbacks to stdout/stderr while honouring the quiet flag.
    reporter = _ConsoleReporter(quiet=bool(args.quiet))
    try:
        outputs = transcribe_file(
            args.input,
            output_dir=args.output_dir,
            formats=args.formats,
            include_timestamps=args.timestamps,
            language=args.language,
            model_size=args.model,
            device=args.device,
            use_correction=False if args.no_correction else None,
            config_path=args.config,
            lmstudio_base_url=args.lmstudio_url,
            lmstudio_model=args.lmstudio_model,
            lmstudio_api_key=args.lmstudio_api_key,
            progress_callback=reporter.progress,
            log_callback=reporter.log,
        )
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nОперация прервана пользователем.", file=sys.stderr)
        return 130
    except Exception as exc:  # pragma: no cover - защита от непредвиденных ошибок
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print("Готово. Созданы файлы:")
        for path in outputs:
            print(f" - {path}")
    else:
        for path in outputs:
            print(path)
    return 0


def transcribe_file(
    input_path: Path,
    *,
    output_dir: Optional[Path] = None,
    formats: Optional[List[str]] = None,
    include_timestamps: Optional[bool] = None,
    language: Optional[str] = None,
    model_size: Optional[str] = None,
    device: Optional[str] = None,
    use_correction: Optional[bool] = None,
    config_path: Optional[Path] = None,
    lmstudio_base_url: Optional[str] = None,
    lmstudio_model: Optional[str] = None,
    lmstudio_api_key: Optional[str] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
    log_callback: Optional[Callable[[str, str], None]] = None,
) -> List[Path]:
    """Fire up a lightweight worker instance and block until a single file is processed."""
    from PyQt6.QtCore import QCoreApplication, QEventLoop

    input_path = Path(input_path).expanduser().resolve()
    # Give users an actionable error rather than letting the worker crash on a missing file.
    if not input_path.exists():
        raise FileNotFoundError(f"Файл '{input_path}' не найден.")

    # Persisted settings form the baseline before layering CLI overrides on top.
    config = _load_config(config_path)
    settings = _build_processing_settings(config)

    # Map explicit CLI flags for the LM Studio integration onto processing settings.
    if lmstudio_base_url is not None:
        settings.lmstudio_base_url = lmstudio_base_url
    if lmstudio_model is not None:
        settings.lmstudio_model = lmstudio_model
    if lmstudio_api_key is not None:
        settings.lmstudio_api_key = lmstudio_api_key
    if settings.lmstudio_base_url and settings.lmstudio_model:
        settings.use_lmstudio = True
    elif lmstudio_base_url is not None or lmstudio_model is not None:
        settings.use_lmstudio = False

    # Resolve where to place outputs, defaulting to the configured directory or source folder.
    target_output_dir = Path(output_dir) if output_dir else Path(config.get("output_dir") or input_path.parent)
    target_output_dir = target_output_dir.expanduser().resolve()
    target_output_dir.mkdir(parents=True, exist_ok=True)

    # Normalise runtime parameters, falling back to config defaults when CLI flags are omitted.
    resolved_language = language or str(config.get("language") or "auto")
    resolved_model = model_size or str(config.get("model_size") or "base")
    resolved_device = (device or str(config.get("device") or "auto")).lower()
    if resolved_device not in {"cpu", "cuda", "hybrid", "auto"}:
        resolved_device = "cpu"

    # Speaker diarisation and correction flags inherit from config unless explicitly toggled.
    resolved_use_correction = (
        bool(config.get("use_local_llm_correction")) if use_correction is None else bool(use_correction)
    )
    resolved_formats = _resolve_formats(config, formats)

    # Determine whether plaintext transcripts should include timestamps.
    timestamps_flag = include_timestamps
    if timestamps_flag is None:
        timestamps_flag = bool(config.get("include_txt_timestamps"))

    # Expand requested formats into OutputRequest objects consumed by the worker.
    requests = _build_output_requests(resolved_formats, timestamps_flag)
    primary_format = requests[0].format if requests else (resolved_formats[0] if resolved_formats else "srt")

    # Package the resolved options into a single transcription task for the background worker.
    task = TranscriptionTask(
        video_path=input_path,
        output_dir=target_output_dir,
        output_format=primary_format,
        language=resolved_language,
        model_size=resolved_model,
    )
    task.device = resolved_device
    task.use_local_llm_correction = resolved_use_correction
    task.deep_correction = settings.deep_correction and task.use_local_llm_correction
    task.local_llm_model_path = str(config.get("local_llm_model_path") or "")
    backend = str(config.get("whisper_backend") or "faster").strip().lower()
    if backend not in {"faster", "openai"}:
        backend = "faster"
    task.whisper_backend = backend
    task.faster_whisper_compute_type = str(config.get("faster_whisper_compute_type") or "int8")
    task.include_timestamps = timestamps_flag
    task.correction_device = str(config.get("correction_device") or settings.correction_device)
    task.outputs = requests
    task.pipeline_mode = settings.pipeline_mode
    task.max_parallel_transcriptions = settings.max_parallel_transcriptions
    task.max_parallel_corrections = settings.max_parallel_corrections
    task.use_lmstudio = settings.use_lmstudio and task.use_local_llm_correction
    task.lmstudio_base_url = settings.lmstudio_base_url
    task.lmstudio_model = settings.lmstudio_model
    task.lmstudio_api_key = settings.lmstudio_api_key
    task.lmstudio_batch_size = settings.lmstudio_batch_size
    task.lmstudio_prompt_token_limit = settings.lmstudio_prompt_token_limit
    task.lmstudio_response_token_limit = settings.lmstudio_response_token_limit
    task.lmstudio_token_margin = settings.lmstudio_token_margin
    task.lmstudio_load_timeout = settings.lmstudio_load_timeout
    task.lmstudio_poll_interval = settings.lmstudio_poll_interval

    # Fallback to offline correction if either LM Studio endpoint detail is missing.
    if not task.lmstudio_base_url or not task.lmstudio_model:
        task.use_lmstudio = False

    # Reuse an existing Qt application when invoked from a GUI process; otherwise create one.
    app = QCoreApplication.instance()
    owns_app = False
    if app is None:
        app = QCoreApplication([])
        owns_app = True

    # Spin up the background worker and shared containers used by the signal callbacks below.
    worker = TranscriptionWorker()
    done = threading.Event()
    result: List[Path] = []
    error_holder: List[str] = []

    # Fan out worker signals to user-provided callbacks so the CLI can surface progress.
    if progress_callback is not None:
        worker.progress_updated.connect(lambda _tid, value: progress_callback(value))
    if log_callback is not None:
        worker.log_message.connect(lambda level, message: log_callback(level, message))

    worker.task_completed.connect(lambda _tid, paths: _on_complete(paths, result, done))
    worker.task_failed.connect(lambda _tid, message: _on_fail(message, error_holder, done))

    worker.update_processing_settings(settings)
    worker.start()
    worker.add_task(task)

    try:
        while not done.wait(0.1):
            # Periodically spin the Qt event loop so queued signals are dispatched to the callbacks above.
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
    finally:
        # Ensure the background worker is torn down cleanly and the Qt app exits when owned here.
        worker.stop()
        worker.wait()
        if owns_app:
            app.quit()

    if error_holder:
        raise RuntimeError(error_holder[0])
    return result


def _load_config(config_path: Optional[Path]) -> AppConfig:
    """Resolve the effective config path (default or CLI override) and load settings."""
    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    return AppConfig(path)


def _build_processing_settings(config: AppConfig) -> ProcessingSettings:
    """Map persisted configuration fields into ProcessingSettings consumed by the worker."""
    return ProcessingSettings(
        pipeline_mode=str(config.get("parallel_mode") or "balanced"),
        max_parallel_transcriptions=int(config.get("max_parallel_transcriptions") or 1),
        max_parallel_corrections=int(config.get("max_parallel_corrections") or 1),
        correction_device=str(config.get("correction_device") or "auto"),
        correction_gpu_layers=int(config.get("llama_gpu_layers") or 0),
        correction_batch_size=int(config.get("correction_batch_size") or 40),
        release_whisper_after_batch=bool(config.get("release_whisper_after_batch")),
        deep_correction=bool(config.get("deep_correction_enabled")),
        llama_n_ctx=int(config.get("llama_n_ctx") or 4096),
        llama_batch_size=int(config.get("llama_batch_size") or 40),
        llama_gpu_layers=int(config.get("llama_gpu_layers") or 0),
        llama_main_gpu=int(config.get("llama_main_gpu") or 0),
        enable_cudnn_benchmark=bool(config.get("enable_cudnn_benchmark")),
        use_lmstudio=bool(config.get("lmstudio_enabled")),
        lmstudio_base_url=str(config.get("lmstudio_base_url") or ""),
        lmstudio_model=str(config.get("lmstudio_model") or ""),
        lmstudio_api_key=str(config.get("lmstudio_api_key") or ""),
        lmstudio_batch_size=int(config.get("lmstudio_batch_size") or 40),
        lmstudio_prompt_token_limit=int(config.get("lmstudio_prompt_token_limit") or 8192),
        lmstudio_response_token_limit=int(config.get("lmstudio_response_token_limit") or 4096),
        lmstudio_token_margin=int(config.get("lmstudio_token_margin") or 512),
        lmstudio_load_timeout=float(config.get("lmstudio_load_timeout") or 600),
        lmstudio_poll_interval=float(config.get("lmstudio_poll_interval") or 1.5),
    )


def _resolve_formats(config: AppConfig, formats: Optional[List[str]]) -> List[str]:
    if formats:
        return [fmt.strip().lower() for fmt in formats if fmt.strip()]
    configured = config.get("output_formats_multi") or []
    if isinstance(configured, list) and configured:
        return [str(fmt).strip().lower() for fmt in configured if str(fmt).strip()]
    fallback = str(config.get("output_format") or "srt")
    return [fallback.lower()]


def _build_output_requests(formats: List[str], include_timestamps: bool) -> List[OutputRequest]:
    """Expand CLI format strings into OutputRequest instances with timestamp preferences."""
    requests: List[OutputRequest] = []
    for fmt in formats or ["srt"]:
        normalized = fmt.strip().lower()
        if normalized in {"txt_ts", "txt_timestamps"}:
            requests.append(OutputRequest(format="txt_ts", include_timestamps=True))
        elif normalized in {"txt", "txt_plain"}:
            target_fmt = "txt_ts" if include_timestamps else "txt_plain"
            requests.append(OutputRequest(format=target_fmt, include_timestamps=include_timestamps))
        else:
            requests.append(OutputRequest(format=normalized, include_timestamps=include_timestamps))
    return requests


def _on_complete(paths, result: List[Path], done: threading.Event):
    """Capture success results from the worker's Qt signal and notify waiters."""
    result[:] = [Path(p).resolve() for p in paths]
    done.set()


def _on_fail(message: str, error_holder: List[str], done: threading.Event):
    """Record error information emitted by the worker and release blocking waits."""
    error_holder[:] = [message]
    done.set()


class _ConsoleReporter:
    """Minimal logger/progress proxy that mirrors worker updates to stdout when not quiet."""
    def __init__(self, quiet: bool):
        self.quiet = quiet
        self._last_progress = -1

    def log(self, level: str, message: str):
        if self.quiet:
            return
        print(f"[{level.upper()}] {message}")

    def progress(self, value: int):
        if self.quiet:
            return
        # Avoid logging the same percentage repeatedly when the worker emits frequent updates.
        if value == self._last_progress:
            return
        self._last_progress = value
        print(f"Прогресс: {value}%")
