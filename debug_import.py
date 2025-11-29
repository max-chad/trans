import sys
import traceback

try:
    print("Attempting to import app.diarization...")
    import app.diarization
    print("Import successful!")
except Exception:
    traceback.print_exc()
