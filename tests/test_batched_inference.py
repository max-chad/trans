import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Mock faster_whisper before importing worker
sys.modules["faster_whisper"] = MagicMock()
from faster_whisper import WhisperModel, BatchedInferencePipeline

from app.worker import TranscriptionWorker
from app.device import DeviceContext
from app.models import TranscriptionTask

class TestBatchedInference(unittest.TestCase):
    def setUp(self):
        self.worker = TranscriptionWorker()
        self.worker.log_message = MagicMock()
        self.worker.model_loaded = MagicMock()
        self.context = DeviceContext("cuda")
        self.context.lock = MagicMock()

    @patch("app.worker.torch.cuda.is_available", return_value=True)
    def test_load_faster_whisper_batched(self, mock_cuda):
        # Setup mocks
        mock_model = MagicMock()
        mock_pipeline = MagicMock()
        
        WhisperModel.return_value = mock_model
        BatchedInferencePipeline.return_value = mock_pipeline
        
        # Call _load_faster_whisper with batched=True
        result = self.worker._load_faster_whisper(
            self.context, 
            "tiny", 
            "float16", 
            batched=True, 
            batch_size=32
        )
        
        # Verify
        self.assertTrue(result)
        BatchedInferencePipeline.assert_called_once_with(model=mock_model)
        self.assertEqual(self.context.model, mock_pipeline)
        self.assertEqual(self.context.backend, "faster_batched")
        self.assertEqual(self.context.compute_type, "float16")

    @patch("app.worker.torch.cuda.is_available", return_value=True)
    def test_load_faster_whisper_sequential(self, mock_cuda):
        # Setup mocks
        mock_model = MagicMock()
        WhisperModel.return_value = mock_model
        BatchedInferencePipeline.reset_mock()
        
        # Call _load_faster_whisper with batched=False
        result = self.worker._load_faster_whisper(
            self.context, 
            "tiny", 
            "float16", 
            batched=False
        )
        
        # Verify
        self.assertTrue(result)
        BatchedInferencePipeline.assert_not_called()
        self.assertEqual(self.context.model, mock_model)
        self.assertEqual(self.context.backend, "faster")

    def test_transcribe_batched(self):
        # Setup context with batched pipeline
        mock_pipeline = MagicMock()
        self.context.model = mock_pipeline
        self.context.backend = "faster_batched"
        
        # Mock transcribe return
        mock_seg = MagicMock()
        mock_seg.start = 0.0
        mock_seg.end = 1.0
        mock_seg.text = "test"
        mock_seg.words = []
        mock_pipeline.transcribe.return_value = ([mock_seg], None)
        
        # Call _transcribe
        self.worker._transcribe(
            self.context, 
            Path("test.wav"), 
            "en", 
            batch_size=24
        )
        
        # Verify call args
        mock_pipeline.transcribe.assert_called_once()
        call_args = mock_pipeline.transcribe.call_args
        self.assertEqual(call_args[0][0], "test.wav")
        self.assertEqual(call_args[1]["batch_size"], 24)
        self.assertNotIn("vad_filter", call_args[1]) # Batched pipeline doesn't use vad_filter arg in same way usually, or we didn't pass it

    def test_transcribe_sequential(self):
        # Setup context with standard model
        mock_model = MagicMock()
        self.context.model = mock_model
        self.context.backend = "faster"
        
        # Mock transcribe return
        mock_seg = MagicMock()
        mock_seg.start = 0.0
        mock_seg.end = 1.0
        mock_seg.text = "test"
        mock_seg.words = []
        mock_model.transcribe.return_value = ([mock_seg], None)
        
        # Call _transcribe
        self.worker._transcribe(
            self.context, 
            Path("test.wav"), 
            "en"
        )
        
        # Verify call args
        mock_model.transcribe.assert_called_once()
        call_args = mock_model.transcribe.call_args
        self.assertEqual(call_args[0][0], "test.wav")
        self.assertTrue(call_args[1]["vad_filter"])
        self.assertNotIn("batch_size", call_args[1])

if __name__ == "__main__":
    unittest.main()
