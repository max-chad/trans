"""Utilities for managing GPU memory and OOM fallbacks."""
from __future__ import annotations

import contextlib
import gc
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class VRAMLimitedModel(contextlib.AbstractContextManager):
    """Context manager that loads a model lazily and frees GPU memory on exit."""

    def __init__(
        self,
        load_fn: Callable[[], Any],
        unload_fn: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self._load_fn = load_fn
        self._unload_fn = unload_fn
        self.model: Any = None

    def __enter__(self) -> Any:  # type: ignore[override]
        logger.debug("Loading VRAM limited model via %s", self._load_fn)
        self.model = self._load_fn()
        return self.model

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        try:
            if self._unload_fn and self.model is not None:
                self._unload_fn(self.model)
        finally:
            self.model = None
            free_cuda()
        return False


def _is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message or "cublas error" in message


def try_gpu_then_cpu(
    fn: Callable[[], Any],
    *,
    on_cpu: Optional[Callable[[], Any]] = None,
    on_fail: Optional[Callable[[BaseException], None]] = None,
) -> Any:
    """Execute ``fn`` and retry on CPU if an OOM error is raised."""

    try:
        return fn()
    except RuntimeError as exc:
        if not _is_oom_error(exc) or on_cpu is None:
            if on_fail:
                on_fail(exc)
            raise
        logger.warning("OOM detected on GPU; retrying on CPU")
        free_cuda()
        return on_cpu()
    except Exception as exc:  # pragma: no cover - unexpected errors
        if on_fail:
            on_fail(exc)
        raise


def free_cuda() -> None:
    """Safely release GPU memory."""

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - torch may be missing
        pass
    finally:
        gc.collect()
