import shutil
import subprocess
from pathlib import Path
from typing import Tuple, Optional, List, Dict

try:
    from moviepy.audio.io.ffmpeg_tools import ffmpeg_extract_audio  # type: ignore
except ModuleNotFoundError:  # moviepy <=1.0.3 compatibility
    try:
        from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio  # type: ignore
    except ModuleNotFoundError:
        ffmpeg_extract_audio = None  # type: ignore
try:
    from moviepy.editor import AudioFileClip, VideoFileClip  # type: ignore
except ModuleNotFoundError:
    AudioFileClip = None  # type: ignore
    VideoFileClip = None  # type: ignore

# Допустимые расширения файлов, из которых можно извлекать аудио.
AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".oga",
    ".opus",
    ".wma",
}

# Minimal gap that should be marked explicitly as silence (in seconds).
SILENCE_GAP_SECONDS = 0.75


def run_ffmpeg_cli(input_path: Path, output_path: Path) -> bool:
    """Запускаем ffmpeg через CLI и конвертируем входной файл в WAV."""
    executable = shutil.which("ffmpeg")
    if not executable:
        return False
    if output_path.exists():
        try:
            output_path.unlink()
        except OSError:
            return False
    cmd = [
        executable,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return result.returncode == 0 and output_path.exists()


def extract_with_ffmpeg(input_path: Path, output_path: Path) -> bool:
    """Пробуем извлечь аудио через moviepy, затем через ffmpeg как резерв."""
    if output_path.exists():
        try:
            output_path.unlink()
        except OSError:
            return False
    if ffmpeg_extract_audio is not None:
        try:
            ffmpeg_extract_audio(str(input_path), str(output_path))
            return output_path.exists()
        except Exception:
            pass
    return run_ffmpeg_cli(input_path, output_path)


def extract_audio(source_path: Path, log_callback=None) -> Tuple[Path, bool]:
    """Извлекаем WAV-дорожку из видео или аудио-файла, используя ffmpeg и moviepy как резерв."""
    ext = source_path.suffix.lower()
    if ext == ".wav":
        return source_path, False
    temp_audio = source_path.with_name(f"{source_path.stem}_temp_audio.wav")

    if extract_with_ffmpeg(source_path, temp_audio):
        return temp_audio, True

    if ext in AUDIO_EXTENSIONS:
        if AudioFileClip is not None:
            try:
                with AudioFileClip(str(source_path)) as audio:
                    audio.write_audiofile(str(temp_audio), codec="pcm_s16le", logger=None)
                return temp_audio, True
            except Exception as exc:
                if log_callback:
                    log_callback("warning", f"MoviePy failed to export audio from {source_path.name}: {exc}")
        else:
            if log_callback:
                log_callback("warning", "MoviePy is not installed; skipping AudioFileClip fallback.")
        if run_ffmpeg_cli(source_path, temp_audio):
            return temp_audio, True
    else:
        if VideoFileClip is not None:
            try:
                video = VideoFileClip(str(source_path))
                try:
                    audio = video.audio
                    if audio is None:
                        raise ValueError("Video has no audio track.")
                    audio.write_audiofile(str(temp_audio), codec="pcm_s16le", logger=None)
                    return temp_audio, True
                finally:
                    video.close()
            except Exception as exc:
                if log_callback:
                    log_callback("warning", f"MoviePy failed to extract audio track from {source_path.name}: {exc}")
        else:
            if log_callback:
                log_callback("warning", "MoviePy is not installed; skipping VideoFileClip fallback.")
        if run_ffmpeg_cli(source_path, temp_audio):
            return temp_audio, True

    raise RuntimeError(
        f"Failed to extract audio from {source_path.name}. "
        "Install MoviePy or ensure ffmpeg is available in PATH."
    )


def silence_entry(start: float, end: float) -> Optional[dict]:
    """Return a placeholder segment for a silent gap."""
    duration = max(0.0, end - start)
    if duration <= 0:
        return None
    label = "[silence]" if duration < 1.0 else f"[silence {duration:.1f}s]"
    return {
        "start": start,
        "end": end,
        "text": label,
        "speaker": None,
        "is_silence": True,
    }


def build_word_timeline(segments: List[dict]) -> List[dict]:
    """Flatten phrase-level segments into per-word timeline with silence markers."""
    timeline: List[dict] = []
    previous_end = 0.0
    for seg in segments:
        seg_start = float(seg.get("start", 0.0) or 0.0)
        seg_end = float(seg.get("end", seg_start) or seg_start)
        speaker = seg.get("speaker")
        words = seg.get("words") or []
        if words:
            for word in words:
                start = float(word.get("start", seg_start) or seg_start)
                end = float(word.get("end", start) or start)
                gap = start - previous_end
                if gap >= SILENCE_GAP_SECONDS and start > previous_end:
                    silence = silence_entry(previous_end, start)
                    if silence:
                        timeline.append(silence)
                text = (word.get("text") or word.get("word") or "").strip()
                if not text:
                    continue
                timeline.append(
                    {
                        "start": start,
                        "end": end if end > start else start,
                        "text": text,
                        "speaker": speaker,
                        "is_silence": False,
                    }
                )
                previous_end = max(previous_end, end)
            if seg_end - previous_end >= SILENCE_GAP_SECONDS and seg_end > previous_end:
                silence = silence_entry(previous_end, seg_end)
                if silence:
                    timeline.append(silence)
            if seg_end > previous_end:
                previous_end = seg_end
            continue
        if seg_start - previous_end >= SILENCE_GAP_SECONDS and seg_start > previous_end:
            silence = silence_entry(previous_end, seg_start)
            if silence:
                timeline.append(silence)
        text = seg.get("text", "").strip()
        if text:
            timeline.append(
                {
                    "start": seg_start,
                    "end": seg_end if seg_end > seg_start else seg_start,
                    "text": text,
                    "speaker": speaker,
                    "is_silence": False,
                }
            )
        previous_end = max(previous_end, seg_end)
    return timeline


def group_segments_by_speaker(segments: List[dict]) -> List[dict]:
    """Объединяем соседние сегменты с одинаковым говорящим в один блок."""
    grouped: List[dict] = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        speaker = seg.get("speaker")
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", start) or start)
        if grouped and grouped[-1].get("speaker") == speaker:
            grouped[-1]["text"] = (grouped[-1]["text"] + " " + text).strip()
            grouped[-1]["end"] = end
        else:
            grouped.append(
                {
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )
    return grouped
