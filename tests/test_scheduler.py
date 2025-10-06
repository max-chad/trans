import json
from pathlib import Path

from pipeline.scheduler import PipelineScheduler


class DummyTranscriber:
    def __init__(self, config, allow_remote=False):
        self.config = config
        self.allow_remote = allow_remote

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def transcribe(self, media_path: Path, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ]
        with output_path.open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")


class DummyCorrector:
    def __init__(self, config, allow_remote=False):
        self.config = config
        self.allow_remote = allow_remote

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def correct_batch(self, texts):
        return [text.upper() for text in texts]


def test_pipeline_stage_sequence(tmp_path):
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "sample.mp4").write_bytes(b"")

    config = {
        "asr": {"model": "dummy", "device": "cpu", "max_parallel": 1},
        "llm": {"backend": "rules", "batch_size": 2},
        "pipeline": {
            "temp_dir": str(tmp_path / "interim"),
            "output_dir": str(tmp_path / "output"),
        },
        "runtime": {"allow_remote": False},
    }

    scheduler = PipelineScheduler(config, asr_cls=DummyTranscriber, corrector_cls=DummyCorrector)
    scheduler.run_all(videos_dir)

    raw_path = tmp_path / "interim" / "sample.raw.jsonl"
    clean_path = tmp_path / "interim" / "sample.clean.jsonl"
    txt_path = tmp_path / "output" / "sample.txt"
    srt_path = tmp_path / "output" / "sample.srt"

    assert raw_path.exists()
    assert clean_path.exists()
    assert txt_path.exists()
    assert srt_path.exists()

    clean_lines = clean_path.read_text(encoding="utf-8").splitlines()
    assert "HELLO" in clean_lines[0]
    assert "WORLD" in clean_lines[1]
