import time
from pathlib import Path
from typing import List, Optional
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
    result_paths: List[Path] = field(default_factory=list)
    error: Optional[str] = None
    device: str = "cpu"
    use_local_llm_correction: bool = True
    local_llm_model_path: str = ""
    whisper_backend: str = "openai"
    faster_whisper_compute_type: str = "int8"
    include_timestamps: bool = False
    correction_device: str = "auto"
    outputs: List["OutputRequest"] = field(default_factory=list)
    pipeline_mode: str = "balanced"
    max_parallel_transcriptions: int = 1
    max_parallel_corrections: int = 1
    use_lmstudio: bool = True
    lmstudio_base_url: str = ""
    lmstudio_model: str = ""
    lmstudio_api_key: str = ""
    lmstudio_batch_size: int = 40


@dataclass
class OutputRequest:
    format: str
    include_timestamps: bool = False


@dataclass
class ProcessingSettings:
    pipeline_mode: str = "balanced"
    max_parallel_transcriptions: int = 1
    max_parallel_corrections: int = 1
    correction_device: str = "auto"
    correction_gpu_layers: int = 0
    correction_batch_size: int = 40
    release_whisper_after_batch: bool = True
    llama_n_ctx: int = 4096
    llama_batch_size: int = 40
    llama_gpu_layers: int = 0
    llama_main_gpu: int = 0
    enable_cudnn_benchmark: bool = True
    use_lmstudio: bool = True
    lmstudio_base_url: str = ""
    lmstudio_model: str = ""
    lmstudio_api_key: str = ""
    lmstudio_batch_size: int = 40
