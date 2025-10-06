# Transcription Pipeline & Desktop App

## CLI Usage

1. Install dependencies and project modules (GPU optional):
   ```bash
   pip install -r requirements.txt
   ```
2. Copy the sample configuration and adjust it to your environment:
   ```bash
   cp config.example.yaml config.yaml
   ```
3. Run staged processing from the command line:
   ```bash
   python run.py all --inputs ./videos --config config.yaml
   ```

## Desktop Application

1. Install UI dependencies:
   ```bash
   pip install -r code/requirements.txt
   ```
2. Launch the PyQt application:
   ```bash
   python -m code.main
   ```
3. Drop media files into the window, configure ASR/LLM settings, and start processing. Results and intermediate files are stored in directories configured via the GUI.
