import unittest
import sys
from unittest.mock import patch, MagicMock

# Import modules normally first so they get the real torch
try:
    from app.gpu_detection import get_optimal_compute_type
    from app.worker import TranscriptionWorker
    from app.device import DeviceContext
except ImportError:
    # Handle case where dependencies might be missing in test env, though they should be there
    pass

class TestGPUOptimization(unittest.TestCase):
    def setUp(self):
        self.mock_torch = MagicMock()
        self.mock_torch.cuda.is_available.return_value = True
        self.mock_torch.cuda.get_device_capability.return_value = (7, 5)

    def test_get_optimal_compute_type_pascal(self):
        self.mock_torch.cuda.get_device_capability.return_value = (6, 1)
        with patch.dict(sys.modules, {"torch": self.mock_torch}):
            self.assertEqual(get_optimal_compute_type(), "int8")

    def test_get_optimal_compute_type_turing(self):
        self.mock_torch.cuda.get_device_capability.return_value = (7, 5)
        with patch.dict(sys.modules, {"torch": self.mock_torch}):
            self.assertEqual(get_optimal_compute_type(), "float16")

    def test_get_optimal_compute_type_ampere(self):
        self.mock_torch.cuda.get_device_capability.return_value = (8, 6)
        with patch.dict(sys.modules, {"torch": self.mock_torch}):
            self.assertEqual(get_optimal_compute_type(), "float16")
    
    def test_get_optimal_compute_type_ada(self):
        self.mock_torch.cuda.get_device_capability.return_value = (8, 9)
        with patch.dict(sys.modules, {"torch": self.mock_torch}):
            self.assertEqual(get_optimal_compute_type(), "float16")

    def test_get_optimal_compute_type_no_cuda(self):
        self.mock_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": self.mock_torch}):
            self.assertEqual(get_optimal_compute_type(), "int8")

    @patch("app.worker.get_optimal_compute_type")
    @patch("app.worker.TranscriptionWorker._load_faster_whisper")
    def test_ensure_model_resolves_auto(self, mock_load, mock_get_optimal):
        # We don't need to patch torch here because we are mocking get_optimal_compute_type
        worker = TranscriptionWorker()
        worker.log_message = MagicMock()
        
        context = DeviceContext("cuda")
        mock_get_optimal.return_value = "float16"
        mock_load.return_value = True
        
        # Should call _load_faster_whisper with "float16" when compute_type is "auto"
        worker._ensure_model_for_context(context, "faster", "tiny", "auto")
        
        mock_load.assert_called_with(context, "tiny", "float16")

if __name__ == "__main__":
    unittest.main()
