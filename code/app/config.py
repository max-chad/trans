import json
from pathlib import Path
from typing import Any, Dict


class AppConfig:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        config_dir = config_path.parent
        self.defaults: Dict[str, Any] = {
            "output_dir": str(Path.home()),
            "device": "cpu",
            "model_size": "small",
            "output_format": "srt",
            "language": "auto",
            "temp_dir": str(config_dir / "interim"),
            "llm_backend": "rules",
            "llm_model_path": "",
            "llm_quant": "q4",
            "llm_batch_size": 4,
            "llm_max_input_len": 2048,
            "allow_remote": False,
        }
        self.settings = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    settings = json.load(f)
                    self.defaults.update(settings)
                    return self.defaults
            except (json.JSONDecodeError, IOError):
                return self.defaults
        return self.defaults

    def save_config(self):
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=4, ensure_ascii=False)
        except IOError:
            pass

    def get(self, key: str) -> Any:
        return self.settings.get(key, self.defaults.get(key))

    def set(self, key: str, value: Any):
        self.settings[key] = value
        self.save_config()

    # ------------------------------------------------------------------
    def output_directory(self) -> Path:
        return Path(self.get("output_dir")).expanduser()

    def temp_directory(self) -> Path:
        return Path(self.get("temp_dir")).expanduser()

    def build_asr_config(self) -> Dict[str, Any]:
        device = str(self.get("device") or "cpu").lower()
        model_name = self.get("model_size") or "small"
        compute_type = "float16" if device == "cuda" else "int8"
        cfg: Dict[str, Any] = {
            "model": model_name,
            "device": device,
            "compute_type": compute_type,
            "max_parallel": 1,
            "vad": "silero",
        }
        if device == "cuda":
            cfg["gpu_memory_limit"] = 7168
        return cfg

    def build_llm_config(self) -> Dict[str, Any]:
        backend = str(self.get("llm_backend") or "rules")
        cfg = {
            "backend": backend,
            "model_path": self.get("llm_model_path"),
            "quant": self.get("llm_quant"),
            "batch_size": int(self.get("llm_batch_size") or 4),
            "max_input_len": int(self.get("llm_max_input_len") or 2048),
        }
        return cfg

    def runtime_flags(self) -> Dict[str, Any]:
        return {
            "allow_remote": bool(self.get("allow_remote")),
        }
