import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.diarization import DiarizationService

def test_download():
    print("Testing DiarizationService.download_models()...")
    # Force allow_download=True to trigger the logic
    # offline_mode=False is implied if allow_download=True usually, but let's be explicit
    service = DiarizationService(device="cpu", offline_mode=False, allow_download=True)
    
    # Clean up previous attempt if any (optional, but good for testing)
    # But maybe dangerous if user has partial data. Let's just run it.
    # The error was 1314, so it probably failed early.
    
    try:
        service.download_models()
        print("Download completed successfully.")
    except Exception as e:
        print(f"Download failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_download()
