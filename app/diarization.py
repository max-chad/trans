import logging
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import List, Optional, Dict
from sklearn.cluster import AgglomerativeClustering

try:
    from speechbrain.inference.speakers import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except (ImportError, AttributeError) as e:
    SPEECHBRAIN_AVAILABLE = False
    SPEECHBRAIN_ERROR = str(e)
    EncoderClassifier = None

class DiarizationService:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.classifier = None
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        if not SPEECHBRAIN_AVAILABLE:
            raise ImportError(f"SpeechBrain is not available: {SPEECHBRAIN_ERROR}")

        if self.classifier is not None:
            return
        
        try:
            # Use a local directory for the model to avoid re-downloading if possible,
            # though SpeechBrain handles caching in ~/.cache/speechbrain by default.
            # We use the 'spkrec-ecapa-voxceleb' model which is small and effective.
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device}
            )
        except Exception as e:
            self.logger.error(f"Failed to load SpeechBrain model: {e}")
            raise

    def _extract_embedding(self, audio_segment: torch.Tensor, sample_rate: int) -> np.ndarray:
        # SpeechBrain expects (batch, time)
        if len(audio_segment.shape) == 1:
            audio_segment = audio_segment.unsqueeze(0)
        
        # Ensure correct sample rate (ECAPA-VoxCeleb expects 16k)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(self.device)
            audio_segment = resampler(audio_segment)

        embeddings = self.classifier.encode_batch(audio_segment)
        # embeddings shape: (batch, 1, 192) -> squeeze to (192,)
        return embeddings.squeeze().cpu().numpy()

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

        self.load_model()
        
        # Load audio
        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))
            waveform = waveform.to(self.device)
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
                
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}")
            
        return segments
