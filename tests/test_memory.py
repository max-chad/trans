import pytest

from utils.memory import try_gpu_then_cpu


def test_try_gpu_then_cpu_fallback():
    attempts = {"gpu": 0, "cpu": 0}

    def gpu_fn():
        attempts["gpu"] += 1
        raise RuntimeError("CUDA out of memory")

    def cpu_fn():
        attempts["cpu"] += 1
        return "cpu-result"

    result = try_gpu_then_cpu(gpu_fn, on_cpu=cpu_fn)
    assert result == "cpu-result"
    assert attempts["gpu"] == 1
    assert attempts["cpu"] == 1


def test_try_gpu_then_cpu_no_retry_on_non_oom():
    def gpu_fn():
        raise RuntimeError("some other error")

    with pytest.raises(RuntimeError):
        try_gpu_then_cpu(gpu_fn, on_cpu=lambda: None)
