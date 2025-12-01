import torch
import torchaudio
import soundfile
import sys

print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}")
print(f"Torchaudio: {torchaudio.__version__}")
print(f"Soundfile: {soundfile.__version__}")

try:
    print(f"Torchaudio backends: {torchaudio.list_audio_backends()}")
except Exception as e:
    print(f"Error listing backends: {e}")

try:
    import torchcodec
    print(f"Torchcodec: {torchcodec.__version__}")
except ImportError:
    print("Torchcodec not installed")
