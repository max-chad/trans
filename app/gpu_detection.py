from __future__ import annotations

"""Утилиты для определения наличия поддерживаемых видеокарт NVIDIA серии RTX 40/50."""

import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Iterable, List


# Регулярное выражение для распознавания моделей RTX 40xx и 50xx в произвольной записи.
RTX_40_50_PATTERN = re.compile(r"rtx\s*(?:4|5)0\d{2}", re.IGNORECASE)


def _normalize_name(name: str) -> str:
    """Сжимает последовательности пробелов и удаляет пробелы по краям названия GPU."""
    return re.sub(r"\s+", " ", name).strip()


def _matches_supported_series(name: str) -> bool:
    """Проверяет, содержит ли название видеокарты обозначение поддерживаемой серии."""
    if not name:
        return False
    normalized = _normalize_name(name).lower()
    if RTX_40_50_PATTERN.search(normalized):
        return True
    # Handle workstation naming such as "RTX 5000 Ada Generation" or "GeForce RTX 4090D"
    normalized = normalized.replace("-", " ")
    if RTX_40_50_PATTERN.search(normalized):
        return True
    return False


def _deduplicate(names: Iterable[str]) -> List[str]:
    """Удаляет пустые строки и дубликаты, сохраняя исходный порядок названий."""
    seen = set()
    result: List[str] = []
    for raw in names:
        cleaned = raw.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _detect_with_torch() -> List[str]:
    """Вызывает CUDA-API PyTorch для получения списков имен доступных устройств."""
    try:
        import torch  # type: ignore
    except Exception:
        return []
    try:
        if torch.cuda.device_count() == 0:
            return []
        return [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except Exception:
        return []


def _detect_with_nvml() -> List[str]:
    """Использует библиотеку NVML (pynvml) для опроса видеокарт через драйвер NVIDIA."""
    try:
        import pynvml  # type: ignore
    except Exception:
        return []
    try:
        pynvml.nvmlInit()
    except Exception:
        return []
    try:
        count = pynvml.nvmlDeviceGetCount()
        result: List[str] = []
        for index in range(count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="ignore")
                result.append(str(name))
            except Exception:
                continue
        return result
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _detect_with_nvidia_smi() -> List[str]:
    """Читает список видеокарт командой `nvidia-smi`, если утилита доступна в системе."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
    except Exception:
        return []
    if completed.returncode != 0 or not completed.stdout:
        return []
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return lines


@dataclass
class GPUDetectionResult:
    """Результат обнаружения, включающий исходные и отфильтрованные названия GPU."""

    raw_names: List[str]
    matched_names: List[str]
    torch_usable: bool

    @property
    def any_gpu(self) -> bool:
        return bool(self.raw_names)

    @property
    def has_supported_series(self) -> bool:
        return bool(self.matched_names)


def detect_supported_nvidia_gpus() -> GPUDetectionResult:
    """Агрегирует данные из всех доступных источников и выделяет поддерживаемые модели."""
    raw_names: List[str] = []

    torch_names = _detect_with_torch()
    torch_usable = bool(torch_names)
    raw_names.extend(torch_names)

    nvml_names = _detect_with_nvml()
    raw_names.extend(nvml_names)

    smi_names = _detect_with_nvidia_smi()
    raw_names.extend(smi_names)

    unique_names = _deduplicate(raw_names)
    matched_names = [name for name in unique_names if _matches_supported_series(name)]
    return GPUDetectionResult(
        raw_names=unique_names,
        matched_names=matched_names,
        torch_usable=torch_usable,
    )
