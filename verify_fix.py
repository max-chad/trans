import torch
from pathlib import Path
from app.diarization import DiarizationService
import soundfile as sf
import numpy as np

# Create a dummy wav file
dummy_wav = Path("test_audio.wav")
sr = 16000
duration = 1.0
t = np.linspace(0, duration, int(sr * duration))
y = 0.5 * np.sin(2 * np.pi * 440 * t)
sf.write(dummy_wav, y, sr)

print(f"Created dummy audio file: {dummy_wav}")

try:
    service = DiarizationService(device="cpu")
    print("DiarizationService initialized.")
    
    waveform, sample_rate = service._load_waveform(dummy_wav)
    print(f"Loaded waveform shape: {waveform.shape}")
    print(f"Sample rate: {sample_rate}")
    
    if waveform is not None and sample_rate == sr:
        print("SUCCESS: Audio loaded correctly via soundfile path.")
    else:
        print("FAILURE: Audio loading returned unexpected results.")

except Exception as e:
    print(f"FAILURE: Exception occurred: {e}")
finally:
    if dummy_wav.exists():
        dummy_wav.unlink()
        print("Cleaned up dummy audio file.")
