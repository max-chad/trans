import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum


class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"


@dataclass
class TranscriptionTask:
    video_path: Path
    output_dir: Path
    output_format: str
    language: str
    model_size: str
    task_id: str = field(default_factory=lambda: f"task_{int(time.time() * 1000)}")
    status: str = "pending"
    progress: int = 0
    result_path: Optional[Path] = None
    error: Optional[str] = None
    device: str = "cpu"
    use_g4f_correction: bool = True
    g4f_model: str = "gpt-4o-mini"
