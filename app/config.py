import json
import os
import sys
from pathlib import Path
from typing import Any, Dict
import hashlib
import shutil

LEGACY_KEY_MAP = {
    "use_g4f_correction": "use_local_llm_correction",
    "use_g4f_translation": "use_local_llm_translation",
    "g4f_model": "local_llm_model_path",
}

CONFIG_ROOT_DIR = Path.home() / ".video-transcriber"
LEGACY_CONFIG_PATH = CONFIG_ROOT_DIR / "config.json"
_PROJECTS_SUBDIR = "projects"


def _project_root() -> Path:
    # When packaged as an .exe, resolve paths relative to the executable so config/log dirs stay stable.
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    resolved = Path(__file__).resolve()
    parents = resolved.parents
    return parents[1] if len(parents) > 1 else resolved.parent


def _sanitize_project_name(name: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name.lower()).strip("-")
    return sanitized or "project"


def _project_namespace() -> str:
    root = _project_root()
    digest = hashlib.sha1(str(root).encode("utf-8")).hexdigest()[:8]
    return f"{_sanitize_project_name(root.name)}-{digest}"


def get_default_config_dir() -> Path:
    env_override = os.getenv("VIDEO_TRANSCRIBER_CONFIG_DIR")
    if env_override:
        return Path(env_override).expanduser()
    return CONFIG_ROOT_DIR / _PROJECTS_SUBDIR / _project_namespace()


def get_default_config_path() -> Path:
    return get_default_config_dir() / "config.json"


DEFAULT_CONFIG_PATH = get_default_config_path()


class AppConfig:
    """Менеджер настроек: отвечает за чтение, миграцию и сохранение конфигурации."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.load_warning: str | None = None
        self._maybe_migrate_legacy_config()
        self.defaults: Dict[str, Any] = {
            "output_dir": str(Path.home()),
            "device": "hybrid",
            "model_size": "base",
            "window_width": 1200,
            "window_height": 800,
            "window_maximized": False,
            "splitter_sizes": [720, 480],
            "left_splitter_sizes": [420, 580],
            "output_format": "srt",
            "language": "ru",
            "translate_lang": "en",
            "use_local_llm_correction": True,
            "use_local_llm_translation": True,
            "local_llm_model_path": "",
            "whisper_backend": "faster",
            "faster_whisper_compute_type": "float16",
            "include_txt_timestamps": False,
            "post_processing_action": "none",
            "post_processing_notification_text": "Processing complete",
            "parallel_mode": "balanced",
            "output_formats_multi": ["srt"],
            "llama_gpu_layers": 0,
            "llama_main_gpu": 0,
            "llama_batch_size": 40,
            "llama_n_ctx": 4096,
            "release_whisper_after_batch": False,
            "enable_cudnn_benchmark": True,
            "correction_batch_size": 40,
            "deep_correction_enabled": False,
            "lmstudio_enabled": True,
            "lmstudio_base_url": "http://127.0.0.1:1234/v1",
            "lmstudio_model": "google/gemma-3n-e4b",
            "lmstudio_api_key": "",
            "lmstudio_batch_size": 40,
            "lmstudio_prompt_token_limit": 8192,
            "lmstudio_load_timeout": 600,
            "lmstudio_poll_interval": 1.5,
            "enable_diarization": False,
            "diarization_num_speakers": 0,
            "diarization_threshold": 0.8,
            "diarization_device": "auto",
            "diarization_compute_type": "auto",
            "batched_inference_enabled": True,
            "batched_inference_batch_size": 16,
        }
        self.settings = self.load_config()

    def _maybe_migrate_legacy_config(self):
        """Copy legacy shared config into the project-specific location once."""
        if (
            self.config_path == DEFAULT_CONFIG_PATH
            and not self.config_path.exists()
            and LEGACY_CONFIG_PATH.exists()
        ):
            try:
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(LEGACY_CONFIG_PATH, self.config_path)
            except OSError:
                pass

    def _preserve_corrupt_config(self) -> Path | None:
        """Keep a copy of the unreadable config so users can inspect what went wrong."""
        if not self.config_path.exists():
            return None
        suffix = self.config_path.suffix or ".json"
        backup = self.config_path.with_suffix(suffix + ".corrupt")
        counter = 1
        while backup.exists():
            backup = self.config_path.with_suffix(f"{suffix}.corrupt{counter}")
            counter += 1
        try:
            shutil.copy2(self.config_path, backup)
            return backup
        except OSError:
            return None

    def _migrate_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Переименовывает устаревшие ключи и восстанавливает совместимость значений."""
        migrated = dict(settings)
        for old_key, new_key in LEGACY_KEY_MAP.items():
            if old_key in migrated and new_key not in migrated:
                migrated[new_key] = migrated.pop(old_key)
        # Подхватываем значение пути из старого ключа в верхнем регистре.
        legacy_upper_path = migrated.pop("LOCAL_LLM_MODEL_PATH", None)
        current_path = (migrated.get("local_llm_model_path") or "").strip()
        if (not current_path or current_path.lower() == "gpt-4o-mini") and legacy_upper_path:
            migrated["local_llm_model_path"] = legacy_upper_path
        elif current_path.lower() == "gpt-4o-mini":
            migrated["local_llm_model_path"] = ""
        # Если список форматов отсутствует, восстанавливаем его из одиночного значения.
        if "output_formats_multi" not in migrated:
            legacy_format = migrated.get("output_format", self.defaults["output_format"])
            include_ts = bool(migrated.get("include_txt_timestamps", False))
            formats = []
            if legacy_format == "srt":
                formats.append("srt")
            elif legacy_format == "txt":
                formats.append("txt_ts" if include_ts else "txt_plain")
            else:
                formats.append(legacy_format)
            migrated["output_formats_multi"] = formats or list(self.defaults["output_formats_multi"])
        # Принудительно переключаем OpenAI/пустой backend на faster-whisper.
        backend = (migrated.get("whisper_backend") or "").strip().lower()
        if backend in {"", "openai"}:
            migrated["whisper_backend"] = "faster"
            
        # Migrate legacy "int8" compute type to "auto" to enable GPU optimization
        compute_type = (migrated.get("faster_whisper_compute_type") or "").strip().lower()
        if compute_type == "int8":
            migrated["faster_whisper_compute_type"] = "auto"
            
        return migrated

    def load_config(self) -> Dict[str, Any]:
        """Загружает конфигурацию с диска, объединяя её с умолчаниями."""
        self.load_warning = None
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    raw_settings = json.load(f)
                if isinstance(raw_settings, dict):
                    merged = {**self.defaults, **self._migrate_settings(raw_settings)}
                    return merged
                self.load_warning = f"Config at {self.config_path} is not a JSON object. Using defaults."
            except (json.JSONDecodeError, IOError) as exc:
                backup = self._preserve_corrupt_config()
                note = f"Failed to parse config at {self.config_path}: {exc}. Using defaults."
                if backup:
                    note += f" Saved unreadable copy as {backup.name}."
                self.load_warning = note
        return dict(self.defaults)

    def save_config(self):
        """Сохраняет текущие настройки, создавая директории при необходимости."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=4, ensure_ascii=False)
        except IOError:
            pass

    def get(self, key: str) -> Any:
        """Возвращает значение по ключу, приводя типы и обеспечивая устойчивость."""
        if key == "local_llm_model_path":
            value = self.settings.get(key) or os.getenv("local_llm_model_path", "")
            return value
        if key == "output_formats_multi":
            value = self.settings.get(key, self.defaults.get(key, []))
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                return [value]
            return list(self.defaults.get(key, []))
        if key in {"splitter_sizes", "left_splitter_sizes"}:
            default = self.defaults.get(key, [720, 480])
            value = self.settings.get(key, default)
            if isinstance(value, list) and len(value) == 2:
                try:
                    return [int(value[0]), int(value[1])]
                except (TypeError, ValueError):
                    pass
            return list(default)
        if key in {
            "llama_gpu_layers",
            "llama_main_gpu",
            "llama_batch_size",
            "llama_n_ctx",
            "correction_batch_size",
            "lmstudio_batch_size",
            "lmstudio_prompt_token_limit",
            "lmstudio_load_timeout",
            "window_width",
            "window_height",
            "diarization_num_speakers",
        }:
            try:
                return int(self.settings.get(key, self.defaults.get(key, 0)))
            except (TypeError, ValueError):
                return int(self.defaults.get(key, 0))
        if key in {"lmstudio_poll_interval", "diarization_threshold"}:
            try:
                return float(self.settings.get(key, self.defaults.get(key, 0.0)))
            except (TypeError, ValueError):
                return float(self.defaults.get(key, 0.0))
        if key in {
            "release_whisper_after_batch",
            "enable_cudnn_benchmark",
            "lmstudio_enabled",
            "window_maximized",
            "deep_correction_enabled",
            "enable_diarization",
        }:
            return bool(self.settings.get(key, self.defaults.get(key, False)))
        return self.settings.get(key, self.defaults.get(key))

    def set(self, key: str, value: Any):
        """Обновляет настройку и сразу же сохраняет конфигурацию на диск."""
        self.settings[key] = value
        self.save_config()
