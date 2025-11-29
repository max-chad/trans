import sys
import traceback

try:
    print("Attempting to import speechbrain...")
    import speechbrain
    from speechbrain.inference.speakers import EncoderClassifier
    print("SpeechBrain import successful!")
except Exception:
    traceback.print_exc()
