import logging
import gc
import os
import shutil
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import AgglomerativeClustering
from inspect import signature

# Compatibility shim: newer huggingface_hub removed `use_auth_token`, but
# SpeechBrain still passes it. Provide a wrapper so both signatures work.
try:
    import huggingface_hub

    _hf_download_sig = signature(huggingface_hub.hf_hub_download)
    if "use_auth_token" not in _hf_download_sig.parameters:
        _orig_hf_hub_download = huggingface_hub.hf_hub_download

        def _hf_hub_download_compat(*args, use_auth_token=None, token=None, **kwargs):
            if use_auth_token is not None and token is None:
                token = use_auth_token
            return _orig_hf_hub_download(*args, token=token, **kwargs)

        huggingface_hub.hf_hub_download = _hf_hub_download_compat  # type: ignore[attr-defined]
        try:
            from huggingface_hub import file_download as _hf_file_download

            _hf_file_download.hf_hub_download = _hf_hub_download_compat  # type: ignore[attr-defined]
        except Exception:
            pass
except Exception:
    huggingface_hub = None

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

    def __init__(self, device: str = "auto", compute_type: str = "auto",
                 offline_mode: bool = True, allow_download: bool = False):
        """
        Args:
            device: Preferred device ('cpu', 'cuda', 'auto', or explicit CUDA id like 'cuda:0').
            compute_type: Precision ('auto', 'float16', 'float32', 'bfloat16').
            offline_mode: If True (default), prevents network access for model downloads.
            allow_download: If True, overrides offline_mode to allow model downloads.
        """
        self.logger = logging.getLogger(__name__)
        self.requested_device = (device or "auto").lower()
        self.compute_type = (compute_type or "auto").lower()
        self.device = self._resolve_device(self.requested_device)
        self.classifier = None
        self._model_device: Optional[str] = None
        self._resamplers: Dict[Tuple[int, int, str], torchaudio.transforms.Resample] = {}
        # Offline if offline_mode=True and allow_download=False.
        # allow_download overrides offline_mode when True.
        # Also respect the environment variable as a fallback.
        env_allow = _env_flag("SPEECHBRAIN_ALLOW_DOWNLOAD", False)
        self.offline = offline_mode and not allow_download and not env_allow
        self.repo_root = Path(__file__).resolve().parent.parent
        self.savedir = self.repo_root / "pretrained_models" / "spkrec-ecapa-voxceleb"
        try:
            torchaudio.set_audio_backend("soundfile")
        except Exception:
            # Not all builds expose set_audio_backend (or the soundfile backend), but we still fall back below.
            pass

    def _materialize_existing_model_files(self) -> None:
        """
        Replace hub-created symlinks with real files to avoid Windows file-lock errors
        when copying/downloading into the target directory.
        """
        candidates = self.REQUIRED_FILES + self.OPTIONAL_LABEL_FILES + ["custom.py"]
        for name in candidates:
            path = self.savedir / name
            if not path.exists() or not path.is_symlink():
                continue
            try:
                target = path.resolve()
                tmp = path.with_suffix(path.suffix + ".tmpcopy")
                shutil.copy2(target, tmp)
                path.unlink(missing_ok=True)
                tmp.replace(path)
                self.logger.info(f"Replaced symlink with local copy: {name}")
            except Exception as exc:
                self.logger.warning(f"Could not replace symlink {path}: {exc}")

    def _ensure_custom_module(self) -> Path:
        """
        SpeechBrain tries to fetch 'custom.py' even when the upstream repo does not ship one.
        Create a local placeholder to short-circuit that fetch and avoid 404 errors.
        """
        self.savedir.mkdir(parents=True, exist_ok=True)
        custom_py = self.savedir / "custom.py"
        if custom_py.exists():
            return custom_py
        try:
            custom_py.write_text(
                "# Placeholder to satisfy SpeechBrain custom module fetch; upstream repo has no custom.py\n",
                encoding="utf-8",
            )
            self.logger.info("Created placeholder custom.py to bypass missing file on the hub.")
        except Exception as exc:
            self.logger.warning(f"Unable to create placeholder custom.py: {exc}")
        return custom_py

    def download_models(self) -> None:
        """
        Explicitly download model files to the local directory.
        This allows fully offline usage later.
        """
        if not SPEECHBRAIN_AVAILABLE:
             raise ImportError(f"SpeechBrain is not available: {SPEECHBRAIN_ERROR}")

        self.logger.info(f"Downloading SpeechBrain models to {self.savedir}...")
        self.savedir.mkdir(parents=True, exist_ok=True)
        # Ensure we have a local custom.py so SpeechBrain doesn't try to fetch a non-existent one.
        self._ensure_custom_module()
        # Convert any existing symlinks to real files before copying over them.
        self._materialize_existing_model_files()
        
        try:
            # We must fetch all required files
            files_to_fetch = self.REQUIRED_FILES + self.OPTIONAL_LABEL_FILES
            
            for filename in files_to_fetch:
                try:
                    self.logger.info(f"Fetching {filename}...")
                    
                    # 1. Download to HF cache (no symlinks involved yet if we just get the path)
                    # Use the compat shim we defined or the imported one
                    cached_path = huggingface_hub.hf_hub_download(
                        repo_id=self.MODEL_ID,
                        filename=filename,
                        use_auth_token=None
                    )
                    
                    # 2. Copy to target directory (bypassing symlink creation)
                    target_path = self.savedir / filename
                    if target_path.is_symlink():
                        target_path.unlink(missing_ok=True)
                    tmp_path = target_path.with_suffix(target_path.suffix + ".tmpcopy")
                    shutil.copy2(cached_path, tmp_path)
                    tmp_path.replace(target_path)
                    
                except Exception as e:
                    if filename in self.OPTIONAL_LABEL_FILES:
                        self.logger.warning(f"Optional file {filename} could not be downloaded: {e}")
                    else:
                        raise e
                        
            self.logger.info("All diarization models downloaded successfully.")
            
        except ImportError:
             self.logger.info("Using EncoderClassifier load to trigger download...")
             original_offline = self.offline
             self.offline = False
             os.environ["SPEECHBRAIN_ALLOW_DOWNLOAD"] = "1"
             
             try:
                 self.load_model()
                 self.unload_model()
             finally:
                 self.offline = original_offline
                 if self.offline:
                     os.environ["SPEECHBRAIN_ALLOW_DOWNLOAD"] = "0"


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
            # Avoid fetching a non-existent custom.py from the hub.
            self._ensure_custom_module()
            # Replace any lingering symlinks with real files to avoid copy failures.
            self._materialize_existing_model_files()

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
            
            # Apply compute type casting ONLY to the embedding model
            # The feature extractor (compute_features) must remain in float32
            if self.compute_type == "float16":
                self.classifier.mods.embedding_model.to(dtype=torch.float16)
            elif self.compute_type == "bfloat16":
                self.classifier.mods.embedding_model.to(dtype=torch.bfloat16)
            elif self.compute_type == "float32":
                self.classifier.mods.embedding_model.to(dtype=torch.float32)
                
            self._model_device = self.device
        except Exception as e:
            self.logger.error(f"Failed to load SpeechBrain model: {e}")
            raise

    def unload_model(self):
        """Unload the model and free GPU memory."""
        if self.classifier is not None:
            del self.classifier
            self.classifier = None
        self._resamplers.clear()
        self._model_device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Diarization model unloaded and memory freed.")

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
            # Determine target dtype
            target_dtype = torch.float32
            if self.compute_type == "float16":
                target_dtype = torch.float16
            elif self.compute_type == "bfloat16":
                target_dtype = torch.bfloat16
            
            # 1. Compute features (always float32)
            # We must disable autocast for this step to ensure Fbank works correctly
            with torch.autocast(device_type="cuda", enabled=False):
                feats = self.classifier.mods.compute_features(audio_segment)
                feats = self.classifier.mods.mean_var_norm(feats, torch.tensor([1.0], device=self.device))
            
            # 2. Cast features to target dtype
            if target_dtype != torch.float32:
                feats = feats.to(dtype=target_dtype)
                
            # 3. Run embedding model
            # If auto, let pytorch handle it. If explicit, we already cast the model and input.
            if self.compute_type == "auto":
                 with torch.autocast(device_type="cuda", enabled=self.device.startswith("cuda")):
                     embeddings = self.classifier.mods.embedding_model(feats)
            else:
                 # Manual mode: model and input are already in target_dtype
                 with torch.autocast(device_type="cuda", enabled=False):
                     embeddings = self.classifier.mods.embedding_model(feats)

        # embeddings shape: (batch, 1, 192) -> squeeze to (192,)
        return embeddings.squeeze().detach().cpu().float().numpy()

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
