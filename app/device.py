import os
import sys
import ctypes
import threading
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import TranscriptionTask

_CUDA_PATHS_CONFIGURED = False
_CUDA_DLL_HANDLES: List[object] = []
_CUDA_DLL_PATHS: List[str] = []


def configure_cuda_dll_search_path() -> None:
    """Добавляем CUDA-библиотеки в системные пути, чтобы Whisper мог работать на GPU в Windows."""
    global _CUDA_PATHS_CONFIGURED
    if _CUDA_PATHS_CONFIGURED or os.name != "nt":
        return
    base = Path(sys.prefix) if hasattr(sys, "prefix") else None
    candidates: list[Path] = []
    cuda_root: Optional[Path] = None
    if base:
        site_root = base / "Lib" / "site-packages" / "nvidia"
        cuda_root = site_root
        for subdir in ("cublas", "cudnn", "cuda_runtime"):
            candidates.append(site_root / subdir / "bin")
            candidates.append(site_root / subdir / "lib" / "x64")
        torch_lib = base / "Lib" / "site-packages" / "torch" / "lib"
        candidates.append(torch_lib)
        # Some wheel builds place CUDA DLLs directly under torch root.
        candidates.append(torch_lib.parent)
    env_path = os.environ.get("PATH", "")
    updated_env = env_path
    seen = {part for part in env_path.split(os.pathsep) if part}
    for directory in candidates:
        if not directory.is_dir():
            continue
        directory_str = str(directory)
        try:
            handle = os.add_dll_directory(directory_str)
            _CUDA_DLL_HANDLES.append(handle)
            _CUDA_DLL_PATHS.append(directory_str)
        except AttributeError:
            if directory_str not in seen:
                updated_env = directory_str + os.pathsep + updated_env
                seen.add(directory_str)
                _CUDA_DLL_PATHS.append(directory_str)
    if updated_env != env_path:
        os.environ["PATH"] = updated_env
        # Propagate CUDA_PATH for libraries that inspect it explicitly.
        target_root = cuda_root or base
        if target_root:
            os.environ.setdefault("CUDA_PATH", str(target_root))
    for lib_name in (
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cudart64_12.dll",
        "cusolver64_11.dll",
        "cusparse64_12.dll",
    ):
        try:
            ctypes.WinDLL(lib_name)
        except OSError:
            # Leave detailed logging to the worker when the issue manifests.
            continue
    _CUDA_PATHS_CONFIGURED = True


@dataclass
class DeviceContext:
    """Контекст вычислительного устройства с очередью задач и загруженной моделью."""
    device: str
    queue: "Queue[Optional[TranscriptionTask]]" = field(default_factory=Queue)
    thread: Optional[threading.Thread] = None
    model: Optional[object] = None
    model_size: Optional[str] = None
    backend: Optional[str] = None
    compute_type: Optional[str] = None
    runtime_device: str = "cpu"
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    active: bool = False
