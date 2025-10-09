import json
import os
from pathlib import Path
from typing import Any, Dict

LEGACY_KEY_MAP = {
    "use_g4f_correction": "use_local_llm_correction",
    "use_g4f_translation": "use_local_llm_translation",
    "g4f_model": "local_llm_model_path",
}


class AppConfig:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.defaults: Dict[str, Any] = {
            "output_dir": str(Path.home()),
            "device": "cuda",
            "model_size": "base",
            "output_format": "srt",
            "language": "ru",
            "translate_lang": "en",
            "use_local_llm_correction": True,
            "use_local_llm_translation": True,
            "local_llm_model_path": "C:\\Users\\max_chad\\.lmstudio\\models\\lmstudio-community\\gemma-3n-E4B-it-text-GGUF\\gemma-3n-E4B-it-Q6_K.gguf",
            "whisper_backend": "openai",
            "faster_whisper_compute_type": "int8",
            "include_txt_timestamps": False,
            "post_processing_action": "none",
            "post_processing_notification_text": "Processing complete",
            "parallel_mode": "balanced",
            "max_parallel_transcriptions": 1,
            "max_parallel_corrections": 2,
            "correction_device": "auto",
            "output_formats_multi": ["srt"],
            "llama_gpu_layers": 0,
            "llama_main_gpu": 0,
            "llama_batch_size": 40,
            "llama_n_ctx": 4096,
            "release_whisper_after_batch": True,
            "enable_cudnn_benchmark": True,
            "correction_batch_size": 40,
            "lmstudio_enabled": True,
            "lmstudio_base_url": "http://127.0.0.1:1234/v1",
            "lmstudio_model": "google/gemma-3n-e4b",
            "lmstudio_api_key": "",
            "lmstudio_batch_size": 40,
        }
        self.settings = self.load_config()

    def _migrate_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        migrated = dict(settings)
        for old_key, new_key in LEGACY_KEY_MAP.items():
            if old_key in migrated and new_key not in migrated:
                migrated[new_key] = migrated.pop(old_key)
        legacy_upper_path = migrated.pop("LOCAL_LLM_MODEL_PATH", None)
        current_path = (migrated.get("local_llm_model_path") or "").strip()
        if (not current_path or current_path.lower() == "gpt-4o-mini") and legacy_upper_path:
            migrated["local_llm_model_path"] = legacy_upper_path
        elif current_path.lower() == "gpt-4o-mini":
            migrated["local_llm_model_path"] = ""
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
        return migrated

    def load_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    raw_settings = json.load(f)
                if isinstance(raw_settings, dict):
                    merged = {**self.defaults, **self._migrate_settings(raw_settings)}
                    return merged
            except (json.JSONDecodeError, IOError):
                pass
        return dict(self.defaults)

    def save_config(self):
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=4, ensure_ascii=False)
        except IOError:
            pass

    def get(self, key: str) -> Any:
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
        if key in {
            "max_parallel_transcriptions",
            "max_parallel_corrections",
            "llama_gpu_layers",
            "llama_main_gpu",
            "llama_batch_size",
            "llama_n_ctx",
            "correction_batch_size",
            "lmstudio_batch_size",
        }:
            try:
                return int(self.settings.get(key, self.defaults.get(key, 0)))
            except (TypeError, ValueError):
                return int(self.defaults.get(key, 0))
        if key in {"release_whisper_after_batch", "enable_cudnn_benchmark", "lmstudio_enabled"}:
            return bool(self.settings.get(key, self.defaults.get(key, False)))
        return self.settings.get(key, self.defaults.get(key))

    def set(self, key: str, value: Any):
        self.settings[key] = value
        self.save_config()
