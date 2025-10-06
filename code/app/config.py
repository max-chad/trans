import json
from pathlib import Path
from typing import Dict, Any


class AppConfig:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.defaults: Dict[str, Any] = {
            "output_dir": str(Path.home()),
            "device": "cpu",
            "model_size": "base",
            "output_format": "srt",
            "language": "ru",
            "translate_lang": "en",
            "use_g4f_correction": True,
            "use_g4f_translation": True,
            "g4f_model": "gpt-4o-mini"
        }
        self.settings = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    self.defaults.update(settings)
                    return self.defaults
            except (json.JSONDecodeError, IOError):
                return self.defaults
        return self.defaults

    def save_config(self):
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=4, ensure_ascii=False)
        except IOError:
            pass

    def get(self, key: str) -> Any:
        return self.settings.get(key, self.defaults.get(key))

    def set(self, key: str, value: Any):
        self.settings[key] = value
        self.save_config()
