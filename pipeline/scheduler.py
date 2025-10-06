"""Pipeline scheduler orchestrating staged processing."""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Sequence

from asr.transcriber import ASRTranscriber
from nlp.local_corrector import LocalTextCorrector
from utils.memory import free_cuda

logger = logging.getLogger(__name__)


class PipelineScheduler:
    def __init__(
        self,
        config: Dict[str, Any],
        *,
        asr_cls=ASRTranscriber,
        corrector_cls=LocalTextCorrector,
    ) -> None:
        self.config = config
        self.asr_cls = asr_cls
        self.corrector_cls = corrector_cls
        self.pipeline_cfg = config.get("pipeline", {})
        self.asr_cfg = config.get("asr", {})
        self.llm_cfg = config.get("llm", {})
        runtime_cfg = config.get("runtime", {})
        self.allow_remote = bool(runtime_cfg.get("allow_remote", False))
        self.temp_dir = Path(self.pipeline_cfg.get("temp_dir", "./interim"))
        self.output_dir = Path(self.pipeline_cfg.get("output_dir", "./output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def run_stage(self, stage: str, inputs: Path) -> None:
        stage = stage.upper()
        if stage == "A":
            self._run_stage_a(inputs)
        elif stage == "B":
            self._run_stage_b(inputs)
        elif stage == "C":
            self._run_stage_c(inputs)
        else:
            raise ValueError(f"Unsupported stage {stage}")
        free_cuda()

    def run_all(self, inputs: Path) -> None:
        self.run_stage("A", inputs)
        raw_dir = self.temp_dir
        self.run_stage("B", raw_dir)
        self.run_stage("C", raw_dir)

    # Stage A -----------------------------------------------------------
    def _run_stage_a(self, inputs: Path) -> None:
        videos = self._list_media_files(inputs)
        if not videos:
            logger.warning("Stage A: no input videos in %s", inputs)
            return
        max_workers = int(self.asr_cfg.get("max_parallel", 1))
        if self.asr_cfg.get("device", "cuda") == "cuda":
            max_workers = min(max_workers, 1)
        logger.info("Stage A: transcribing %d files", len(videos))

        def worker(video_path: Path) -> None:
            target = self.temp_dir / f"{video_path.stem}.raw.jsonl"
            with self.asr_cls(self.asr_cfg, allow_remote=self.allow_remote) as transcriber:
                transcriber.transcribe(video_path, target)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(worker, videos))
        free_cuda()

    # Stage B -----------------------------------------------------------
    def _run_stage_b(self, inputs: Path) -> None:
        raw_files = sorted(inputs.glob("*.raw.jsonl"))
        if not raw_files:
            logger.warning("Stage B: no raw transcripts in %s", inputs)
            return
        max_workers = int(self.llm_cfg.get("max_parallel", 1))
        if self.llm_cfg.get("backend", "rules") != "rules":
            max_workers = min(max_workers, 1)
        logger.info("Stage B: correcting %d transcripts", len(raw_files))

        def worker(raw_path: Path) -> None:
            clean_name = raw_path.stem.replace(".raw", "") + ".clean.jsonl"
            clean_path = raw_path.with_name(clean_name)
            segments = self._load_segments(raw_path)
            batch_size = max(1, int(self.llm_cfg.get("batch_size", self.llm_cfg.get("max_batch", 4))))
            with self.corrector_cls(self.llm_cfg, allow_remote=self.allow_remote) as corrector:
                corrected_segments: List[Dict[str, Any]] = []
                for i in range(0, len(segments), batch_size):
                    batch = [seg["text"] for seg in segments[i : i + batch_size]]
                    corrected_texts = corrector.correct_batch(batch)
                    for seg, new_text in zip(segments[i : i + batch_size], corrected_texts):
                        corrected_segments.append({**seg, "text": new_text})
            self._write_segments(clean_path, corrected_segments)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(worker, raw_files))
        free_cuda()

    # Stage C -----------------------------------------------------------
    def _run_stage_c(self, inputs: Path) -> None:
        clean_files = sorted(inputs.glob("*.clean.jsonl"))
        if not clean_files:
            logger.warning("Stage C: no cleaned transcripts in %s", inputs)
            return
        logger.info("Stage C: exporting %d transcripts", len(clean_files))
        for path in clean_files:
            segments = self._load_segments(path)
            self._export_to_txt(path.stem.replace(".clean", ""), segments)
            self._export_to_srt(path.stem.replace(".clean", ""), segments)
        free_cuda()

    # Helpers -----------------------------------------------------------
    def _list_media_files(self, directory: Path) -> List[Path]:
        extensions = {".mp3", ".wav", ".mp4", ".mkv", ".mov", ".flac"}
        return [p for p in sorted(directory.iterdir()) if p.suffix.lower() in extensions]

    @staticmethod
    def _load_segments(path: Path) -> List[Dict[str, Any]]:
        segments: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    segments.append(json.loads(line))
        return segments

    @staticmethod
    def _write_segments(path: Path, segments: Sequence[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for seg in segments:
                f.write(json.dumps(seg, ensure_ascii=False) + "\n")

    def _export_to_txt(self, base_name: str, segments: Sequence[Dict[str, Any]]) -> None:
        text = " ".join(seg.get("text", "") for seg in segments)
        target = self.output_dir / f"{base_name}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text.strip() + "\n", encoding="utf-8")

    def _export_to_srt(self, base_name: str, segments: Sequence[Dict[str, Any]]) -> None:
        target = self.output_dir / f"{base_name}.srt"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            for idx, seg in enumerate(segments, start=1):
                start = self._format_timestamp(seg.get("start", 0.0))
                end = self._format_timestamp(seg.get("end", 0.0))
                f.write(f"{idx}\n{start} --> {end}\n{seg.get('text', '').strip()}\n\n")

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        ms = int(round(seconds * 1000))
        hours, rem = divmod(ms, 3600_000)
        minutes, rem = divmod(rem, 60_000)
        secs, ms = divmod(rem, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
