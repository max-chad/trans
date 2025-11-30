import logging
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import AgglomerativeClustering

try:
    from speechbrain.inference.speakers import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except (ImportError, AttributeError) as e:
    SPEECHBRAIN_AVAILABLE = False
    SPEECHBRAIN_ERROR = str(e)
    EncoderClassifier = None


class DiarizationService:
    TARGET_SAMPLE_RATE = 16000

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
            # Use a local directory for the model to avoid re-downloading if possible,
            # though SpeechBrain handles caching in ~/.cache/speechbrain by default.
            # We use the 'spkrec-ecapa-voxceleb' model which is small and effective.
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device}
            )
            self._model_device = self.device
        except Exception as e:
            self.logger.error(f"Failed to load SpeechBrain model: {e}")
            raise

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

    def run(self, audio_path: Path, segments: List[dict], num_speakers: Optional[int] = None) -> List[dict]:
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
        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))
            waveform = waveform.to(self.device, non_blocking=True)
            # If stereo, mix to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        except Exception as e:
            self.logger.error(f"Failed to load audio {audio_path}: {e}")
            return segments

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
        
        # Clustering
        # If num_speakers is not provided, we can try to estimate or use a threshold.
        # AgglomerativeClustering with distance_threshold requires n_clusters=None.
        # A threshold of 0.7-0.8 is often used for cosine distance, but sklearn uses euclidean by default for ward.
        # For cosine, we need metric='cosine' and linkage='average' or 'complete'.
        
        try:
            if num_speakers:
                clustering = AgglomerativeClustering(n_clusters=num_speakers)
            else:
                # Tuning threshold might be needed. 
                # SpeechBrain embeddings are normalized, so cosine distance is appropriate.
                # However, AgglomerativeClustering with cosine metric is available in newer sklearn.
                # Fallback to euclidean on normalized vectors is roughly equivalent to cosine.
                clustering = AgglomerativeClustering(
                    n_clusters=None, 
                    distance_threshold=1.5, # Tunable parameter
                    metric='euclidean',
                    linkage='ward'
                )
            
            labels = clustering.fit_predict(X)
            
            # Assign labels
            for idx, label in zip(valid_indices, labels):
                segments[idx]["speaker"] = f"SPEAKER_{label:02d}"
            self.logger.info(f"Diarization assigned {len(valid_indices)} speaker labels (clusters: {len(set(labels))}).")
                
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            
        return segments
