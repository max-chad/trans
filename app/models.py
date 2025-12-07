import uuid
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum


class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    HYBRID = "hybrid"


@dataclass
class TranscriptionTask:
    video_path: Path
    output_dir: Path
    output_format: str
    language: str
    model_size: str
    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex}")
    status: str = "pending"
    progress: int = 0
    result_paths: List[Path] = field(default_factory=list)
    error: Optional[str] = None
    device: str = "cpu"
    use_local_llm_correction: bool = True
    local_llm_model_path: str = ""
    whisper_backend: str = "faster"
    faster_whisper_compute_type: str = "auto"
    backend_fallback_attempted: bool = False
    deep_correction: bool = False
    include_timestamps: bool = False
    outputs: List["OutputRequest"] = field(default_factory=list)
    pipeline_mode: str = "balanced"
    use_lmstudio: bool = True
    lmstudio_base_url: str = ""
    lmstudio_model: str = ""
    lmstudio_api_key: str = ""
    lmstudio_batch_size: int = 40
    lmstudio_prompt_token_limit: int = 8192
    lmstudio_load_timeout: float = 600.0
    lmstudio_poll_interval: float = 1.5
    staged_transcript_path: Optional[Path] = None
    source_root: Optional[Path] = None
    use_source_dir_as_output: bool = False
    enable_diarization: bool = False
    num_speakers: Optional[int] = None
    diarization_device: str = "auto"
    diarization_compute_type: str = "auto"
    batched_inference_enabled: bool = True
    batched_inference_batch_size: int = 16


@dataclass
class OutputRequest:
    format: str
    include_timestamps: bool = False


@dataclass
class ProcessingSettings:
    pipeline_mode: str = "balanced"
    correction_gpu_layers: int = 0
    correction_batch_size: int = 40
    release_whisper_after_batch: bool = False
    deep_correction: bool = False
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
    lmstudio_prompt_token_limit: int = 8192
    lmstudio_load_timeout: float = 600.0
    lmstudio_poll_interval: float = 1.5
    diarization_device: str = "auto"
    diarization_threshold: float = 0.8
    diarization_compute_type: str = "auto"
    batched_inference_enabled: bool = True
    batched_inference_batch_size: int = 16
