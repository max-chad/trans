import logging
import os
import shutil
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import AgglomerativeClustering

# SpeechBrain (1.0.3) still expects torchaudio.list_audio_backends, which was
# removed in torchaudio 2.9+. Provide a tiny compatibility shim so the import
# doesn't explode on newer torchaudio builds.
if not hasattr(torchaudio, "list_audio_backends"):
    def _list_audio_backends() -> list[str]:
        try:
            import soundfile  # noqa: F401
            return ["soundfile"]
        except Exception:
            return []

    torchaudio.list_audio_backends = _list_audio_backends  # type: ignore[attr-defined]

if not hasattr(torchaudio, "set_audio_backend"):
    def _set_audio_backend(backend: str | None) -> None:  # noqa: ARG001
        return None

    torchaudio.set_audio_backend = _set_audio_backend  # type: ignore[attr-defined]

try:
    try:
        # Newer SpeechBrain packages expose EncoderClassifier under speaker.py
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        # Older versions used speakers.py
        from speechbrain.inference.speakers import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    SPEECHBRAIN_AVAILABLE = True
    SPEECHBRAIN_ERROR = None
except (ImportError, AttributeError, OSError) as e:
    SPEECHBRAIN_AVAILABLE = False
    SPEECHBRAIN_ERROR = str(e)
    EncoderClassifier = None
    LocalStrategy = None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class DiarizationService:
    TARGET_SAMPLE_RATE = 16000
    MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
    REQUIRED_FILES = [
        "hyperparams.yaml",
        "embedding_model.ckpt",
        "mean_var_norm_emb.ckpt",
        "classifier.ckpt",
    ]
    OPTIONAL_LABEL_FILES = ["label_encoder.ckpt", "label_encoder.txt"]

    def __init__(self, device: str = "auto"):
        """
        Args:
            device: Preferred device ('cpu', 'cuda', 'auto', or explicit CUDA id like 'cuda:0').
        """
        self.logger = logging.getLogger(__name__)
        self.requested_device = (device or "auto").lower()
        self.device = self._resolve_device(self.requested_device)
        self.classifier = None
        self._model_device: Optional[str] = None
        self._resamplers: Dict[Tuple[int, int, str], torchaudio.transforms.Resample] = {}
        # Allow download by default; set SPEECHBRAIN_ALLOW_DOWNLOAD=0 to force offline.
        self.offline = not _env_flag("SPEECHBRAIN_ALLOW_DOWNLOAD", True)
        self.repo_root = Path(__file__).resolve().parent.parent
        self.savedir = self.repo_root / "pretrained_models" / "spkrec-ecapa-voxceleb"
        try:
            torchaudio.set_audio_backend("soundfile")
        except Exception:
            # Not all builds expose set_audio_backend (or the soundfile backend), but we still fall back below.
            pass

    def _resolve_device(self, requested: str) -> str:
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            self.logger.warning("CUDA requested for diarization but not available. Falling back to CPU.")
            return "cpu"
        return requested

    def load_model(self):
        if not SPEECHBRAIN_AVAILABLE:
            raise ImportError(f"SpeechBrain is not available: {SPEECHBRAIN_ERROR}")

        if self.classifier is not None:
            return

        try:
            # Re-resolve in case hardware availability changed between init and load.
            self.device = self._resolve_device(self.requested_device)
            # Force HF Hub to avoid symlinks on Windows and prefer fully local use.
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1" if self.offline else "0")

            hf_home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
            os.environ.setdefault("HF_HOME", str(hf_home))
            self.savedir.mkdir(parents=True, exist_ok=True)

            if not self._has_local_model():
                copied = self._copy_from_hf_cache(Path(hf_home))
                if not copied or not self._has_local_model():
                    if self.offline:
                        raise FileNotFoundError(
                            f"SpeechBrain model not found locally at {self.savedir}. "
                            "Download the contents of speechbrain/spkrec-ecapa-voxceleb manually into this folder "
                            "or set SPEECHBRAIN_ALLOW_DOWNLOAD=1 to permit fetching."
                        )

            # We use the 'spkrec-ecapa-voxceleb' model which is small and effective.
            kwargs = {
                "source": self.MODEL_ID,
                "savedir": str(self.savedir),
                "run_opts": {"device": self.device},
                "huggingface_cache_dir": str(hf_home),
            }
            if LocalStrategy is not None:
                kwargs["local_strategy"] = LocalStrategy.COPY
            self.classifier = EncoderClassifier.from_hparams(**kwargs)
            self._model_device = self.device
        except Exception as e:
            self.logger.error(f"Failed to load SpeechBrain model: {e}")
            raise

    def _has_local_model(self) -> bool:
        core_ok = all((self.savedir / name).exists() for name in self.REQUIRED_FILES)
        label_ok = any((self.savedir / name).exists() for name in self.OPTIONAL_LABEL_FILES)
        return core_ok and label_ok

    def _copy_from_hf_cache(self, hf_home: Path) -> bool:
        """Try to copy an existing HF cache snapshot into savedir to avoid symlink use."""
        cache_root = hf_home / "hub" / f"models--{self.MODEL_ID.replace('/', '--')}" / "snapshots"
        if not cache_root.exists():
            return False

        snapshots = sorted(
            (p for p in cache_root.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for snapshot in snapshots:
            try:
                self.logger.info(f"Copying SpeechBrain model snapshot from HF cache: {snapshot}")
                shutil.copytree(snapshot, self.savedir, dirs_exist_ok=True)
                return True
            except Exception as copy_err:
                self.logger.warning(f"Failed to copy SpeechBrain snapshot {snapshot}: {copy_err}")
        return False

    def _get_resampler(self, sample_rate: int) -> torchaudio.transforms.Resample:
        """
        Keep a per-device resampler to avoid recreating transforms on every segment.
        """
        key = (sample_rate, self.TARGET_SAMPLE_RATE, self.device)
        resampler = self._resamplers.get(key)
        if resampler is None:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.TARGET_SAMPLE_RATE,
            ).to(self.device)
            self._resamplers[key] = resampler
        return resampler

    def _extract_embedding(self, audio_segment: torch.Tensor, sample_rate: int) -> np.ndarray:
        # SpeechBrain expects (batch, time)
        if len(audio_segment.shape) == 1:
            audio_segment = audio_segment.unsqueeze(0)
        
        # Ensure correct sample rate (ECAPA-VoxCeleb expects 16k)
        if sample_rate != self.TARGET_SAMPLE_RATE:
            resampler = self._get_resampler(sample_rate)
            audio_segment = resampler(audio_segment)

        with torch.inference_mode():
            autocast_enabled = self.device.startswith("cuda")
            with torch.autocast(device_type="cuda", enabled=autocast_enabled):
                embeddings = self.classifier.encode_batch(audio_segment)
        # embeddings shape: (batch, 1, 192) -> squeeze to (192,)
        return embeddings.squeeze().detach().cpu().numpy()

    def _load_waveform(self, audio_path: Path) -> Optional[Tuple[torch.Tensor, int]]:
        """
        Try soundfile first to avoid torchcodec/FFmpeg issues on Windows, then fall back to torchaudio.
        """
        # 1. Try soundfile
        try:
            import soundfile as sf  # type: ignore
            data, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
            waveform = torch.from_numpy(data)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() == 2:
                waveform = waveform.transpose(0, 1).contiguous()
            else:
                self.logger.error(f"Unexpected audio shape from soundfile for {audio_path.name}: {waveform.shape}")
                return None
            return waveform, int(sample_rate)
        except Exception as sf_err:
            self.logger.warning(f"SoundFile failed to read {audio_path.name}: {sf_err}. Falling back to torchaudio.")

        # 2. Fallback to torchaudio
        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))
            return waveform, int(sample_rate)
        except Exception as exc:
            self.logger.error(
                f"torchaudio.load also failed for {audio_path.name}: {exc}."
            )
            return None

    def run(self, audio_path: Path, segments: List[dict], num_speakers: Optional[int] = None, **kwargs) -> List[dict]:
        """
        Assign speaker labels to the provided segments.
        
        Args:
            audio_path: Path to the full audio file.
            segments: List of dicts with 'start', 'end', 'text'.
            num_speakers: Optional number of speakers if known.
            
        Returns:
            List of segments with 'speaker' field populated.
        """
        if not segments:
            return []

        if not SPEECHBRAIN_AVAILABLE:
            self.logger.warning(f"Skipping diarization: SpeechBrain not available ({SPEECHBRAIN_ERROR})")
            return segments

        # Resolve device once more before a potentially long run so we can prefer GPU when it is present.
        resolved = self._resolve_device(self.requested_device)
        if resolved != self.device:
            self.logger.info(f"Switching diarization device to {resolved}.")
            self.device = resolved
            self.classifier = None  # force reload on the new device
            self._resamplers.clear()

        self.load_model()
        if self.device != self._model_device:
            # Model device changed unexpectedly; reload to stay consistent.
            self.classifier = None
            self._resamplers.clear()
            self.load_model()
        self.logger.info(f"Running speaker diarization on {self.device}.")
        
        # Load audio
        loaded = self._load_waveform(audio_path)
        if loaded is None:
            self.logger.error(f"Failed to load audio {audio_path}. Skipping diarization.")
            return segments
        waveform, sample_rate = loaded
        waveform = waveform.to(self.device, non_blocking=True)
        # If stereo, mix to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        embeddings = []
        valid_indices = []

        for i, seg in enumerate(segments):
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            if end <= start:
                continue
            
            # Convert time to samples
            start_frame = int(start * sample_rate)
            end_frame = int(end * sample_rate)
            
            # Extract segment
            if start_frame >= waveform.shape[1]:
                continue
            
            seg_audio = waveform[:, start_frame:end_frame]
            if seg_audio.shape[1] < 160: # Too short (< 10ms)
                continue

            try:
                emb = self._extract_embedding(seg_audio, sample_rate)
                embeddings.append(emb)
                valid_indices.append(i)
            except Exception as e:
                self.logger.warning(f"Failed to extract embedding for segment {i}: {e}")

        if not embeddings:
            return segments

        X = np.array(embeddings)
        self.logger.info(f"Diarization embeddings collected: {len(embeddings)}/{len(segments)} segments.")
        
        if len(embeddings) < 2:
            self.logger.info("Less than 2 segments/embeddings. Assigning all to SPEAKER_00.")
            for idx in valid_indices:
                segments[idx]["speaker"] = "SPEAKER_00"
            return segments

        try:
            if num_speakers:
                clustering = AgglomerativeClustering(n_clusters=num_speakers)
            else:
                # Use Cosine distance with Average linkage for better speaker clustering.
                # Threshold is configurable (default 0.8).
                threshold = kwargs.get("threshold", 0.8)
                clustering = AgglomerativeClustering(
                    n_clusters=None, 
                    distance_threshold=threshold,
                    metric='cosine',
                    linkage='average'
                )
            
            labels = clustering.fit_predict(X)
            
            # Assign labels
            for idx, label in zip(valid_indices, labels):
                segments[idx]["speaker"] = f"SPEAKER_{label:02d}"
            self.logger.info(f"Diarization assigned {len(valid_indices)} speaker labels (clusters: {len(set(labels))}).")
                
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            
        return segments
