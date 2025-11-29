import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import torch

# Test imports
from app.audio import extract_audio, group_segments_by_speaker
from app.device import DeviceContext
from app.diarization import DiarizationService, SPEECHBRAIN_AVAILABLE
from app.worker import TranscriptionWorker

class TestRefactoring(unittest.TestCase):
    def test_audio_imports(self):
        self.assertTrue(callable(extract_audio))
        self.assertTrue(callable(group_segments_by_speaker))

    def test_device_imports(self):
        ctx = DeviceContext(device="cpu")
        self.assertEqual(ctx.device, "cpu")

    def test_diarization_service_init(self):
        service = DiarizationService(device="cpu")
        self.assertEqual(service.device, "cpu")
        self.assertIsNone(service.classifier)

    @unittest.skipIf(not SPEECHBRAIN_AVAILABLE, "SpeechBrain not available")
    @patch("app.diarization.EncoderClassifier")
    def test_diarization_run_mock(self, mock_classifier):
        # Mock the classifier to avoid loading heavy models
        mock_instance = MagicMock()
        mock_classifier.from_hparams.return_value = mock_instance
        # Mock encode_batch to return a dummy embedding
        # Return shape (batch, 1, 192)
        mock_instance.encode_batch.return_value = torch.zeros((1, 1, 192))
        
        service = DiarizationService(device="cpu")
        
        # Mock torchaudio.load
        with patch("app.diarization.torchaudio.load") as mock_load:
            # Return waveform (1, 16000*5) and sample_rate
            mock_load.return_value = (torch.zeros((1, 16000*5)), 16000)
            
            segments = [{"start": 0.0, "end": 2.0, "text": "Hello"}]
            result = service.run(Path("dummy.wav"), segments, num_speakers=1)
            
            self.assertEqual(len(result), 1)
            self.assertIn("speaker", result[0])
            self.assertTrue(result[0]["speaker"].startswith("SPEAKER_"))

    def test_worker_integration(self):
        worker = TranscriptionWorker()
        self.assertTrue(hasattr(worker, "diarization"))
        self.assertIsInstance(worker.diarization, DiarizationService)

if __name__ == "__main__":
    unittest.main()
