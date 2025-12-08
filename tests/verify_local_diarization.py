import os
import sys
import shutil
from pathlib import Path
import unittest

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.diarization import DiarizationService

class TestLocalDiarization(unittest.TestCase):
    def setUp(self):
        # Ensure we start with a clean state for the model if possible
        # For safety, we won't delete the model if it exists, but we will test the flags.
        pass

    def test_offline_mode_enforcement(self):
        """Test that offline_mode=True sets HF_HUB_OFFLINE=1"""
        # Case 1: Offline=True, AllowDownload=False
        service = DiarizationService(offline_mode=True, allow_download=False)
        try:
            service.load_model()
        except FileNotFoundError:
            # Expected if models are missing
            pass
        except ImportError:
            # SpeechBrain might be missing
            pass
        except Exception as e:
            # If models exist, load_model might succeed, but we want to check env vars
            if "SpeechBrain model not found" not in str(e):
                 print(f"Caught unexpected error: {e}")

        # Check env var
        # Note: load_model sets the env var. If it crashed before setting it (unlikely), this might fail.
        # But load_model sets it very early.
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")

    def test_allow_download(self):
        """Test that allow_download=True sets HF_HUB_OFFLINE=0"""
        service = DiarizationService(offline_mode=True, allow_download=True)
        try:
            service.load_model()
        except Exception:
            pass
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "0")

if __name__ == "__main__":
    unittest.main()
