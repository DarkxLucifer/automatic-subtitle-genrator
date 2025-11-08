import argparse
import os
import subprocess
import tempfile
from pathlib import Path
import torch
import whisperx
from autosub.srt_helpers import seconds_to_srt_timestamp

try:
    from pyannote.audio import Pipeline as DiarizationPipeline
    DIARIZATION_AVAILABLE = True
except Exception:
    DIARIZATION_AVAILABLE = False

def extract_audio(input_path: str, out_audio: str):
    subprocess.run(["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", "-vn", out_audio], check=True)

def transcribe_and_align(audio_path: str, model_name: str = "medium", device: str = "cuda"):
    model = whisperx.load_model(model_name, device=device)
    result = model.transcribe(audio_path)
    align_model, metadata = whisperx.load_align_model(result["language"], device="cpu")
    aligned = whisperx.align(result["segments"], align_model, metadata, audio_path, device="cpu")
    return aligned

def write_srt(segments, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = seconds_to_srt_timestamp(seg["start"])
            end = seconds_to_srt_timestamp(seg["end"])
            text = seg.get("text", "").strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

def write_vtt(segments, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start = seconds_to_srt_timestamp(seg["start"]).replace(",", ".")
            end = seconds_to_srt_timestamp(seg["end"]).replace(",", ".")
            text = seg.get("text", "").strip()
            f.write(f"{start} --> {end}\n{text}\n\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--model", default="medium")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    input_path = args.input
    output_dir = Path(input_path).parent
    base = Path(input_path).stem

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "audio.wav"
        print("Extracting audio...")
        extract_audio(input_path, str(audio_path))
        print("Transcribing and aligning...")
        aligned = transcribe_and_align(str(audio_path), args.model, args.device)
        srt_file = output_dir / f"{base}.srt"
        vtt_file = output_dir / f"{base}.vtt"
        print("Writing subtitles...")
        write_srt(aligned["segments"], srt_file)
        write_vtt(aligned["segments"], vtt_file)
        print("✅ Done:", srt_file, vtt_file)

if __name__ == "__main__":
    main()
