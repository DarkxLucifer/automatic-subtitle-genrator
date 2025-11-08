# src/autosub/transcribe_accurate.py
"""
Accurate transcription pipeline (debug-friendly).

- Uses whisperx (faster_whisper backend) for alignment.
- On CPU runners, forces compute_type=float32 to avoid fp16 errors.
- Writes detailed transcribe.log and prints paths/sizes so CI can find outputs.
"""
import argparse
import os
import subprocess
import tempfile
import traceback
from pathlib import Path
import sys
import torch

# whisperx import (may wrap faster_whisper)
import whisperx

from autosub.srt_helpers import seconds_to_srt_timestamp

LOG_PATH = Path("transcribe.log")

def log(msg: str):
    print(msg)
    try:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
    except Exception:
        pass

def extract_audio(input_path: str, out_audio: str):
    log(f"[extract_audio] extracting from: {input_path} -> {out_audio}")
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", "-vn", out_audio]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    log(f"[extract_audio] ffmpeg returncode={res.returncode}")
    if res.returncode != 0:
        log("[extract_audio] ffmpeg stderr:")
        try:
            log(res.stderr.decode("utf-8", errors="replace"))
        except Exception:
            log("<could not decode ffmpeg stderr>")
        raise RuntimeError("ffmpeg failed to extract audio")
    log("[extract_audio] audio extraction OK")

def transcribe_and_align(audio_path: str, model_name: str = "medium", device: str = "cpu", compute_type: str = "float32"):
    """
    Load whisperx (faster_whisper backend) with explicit compute_type and transcribe+align.
    compute_type: "float32" | "float16" | "int8" (int8 depends on ctranslate2 build)
    """
    log(f"[transcribe_and_align] loading model='{model_name}' device='{device}' compute_type='{compute_type}'")
    # whisperx.load_model accepts compute_type and passes down to faster_whisper / ctranslate2
    model = whisperx.load_model(model_name, device=device, compute_type=compute_type)
    log("[transcribe_and_align] whisper model loaded")
    log("[transcribe_and_align] running initial transcription (may take time)...")
    result = model.transcribe(audio_path)
    log("[transcribe_and_align] initial transcription complete")
    log("[transcribe_and_align] loading align model (cpu) ...")
    align_model, metadata = whisperx.load_align_model(result["language"], device="cpu")
    log("[transcribe_and_align] running alignment (cpu) ...")
    aligned = whisperx.align(result["segments"], align_model, metadata, audio_path, device="cpu")
    log("[transcribe_and_align] alignment complete")
    return aligned

def write_srt(segments, path: str):
    log(f"[write_srt] writing srt to: {path}")
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = seconds_to_srt_timestamp(seg["start"])
            end = seconds_to_srt_timestamp(seg["end"])
            text = seg.get("text", "").strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    try:
        log(f"[write_srt] wrote {path} ({os.path.getsize(path)} bytes)")
    except Exception:
        log(f"[write_srt] wrote {path} (size unknown)")

def write_vtt(segments, path: str):
    log(f"[write_vtt] writing vtt to: {path}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start = seconds_to_srt_timestamp(seg["start"]).replace(",", ".")
            end = seconds_to_srt_timestamp(seg["end"]).replace(",", ".")
            text = seg.get("text", "").strip()
            f.write(f"{start} --> {end}\n{text}\n\n")
    try:
        log(f"[write_vtt] wrote {path} ({os.path.getsize(path)} bytes)")
    except Exception:
        log(f"[write_vtt] wrote {path} (size unknown)")

def safe_path_info(p: Path):
    try:
        return f"{p} exists={p.exists()} size={p.stat().st_size if p.exists() else 'n/a'}"
    except Exception as e:
        return f"{p} exists=ERROR ({e})"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input video/audio file")
    parser.add_argument("--model", default="medium", help="Whisper model (tiny, base, small, medium, large)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for model inference")
    parser.add_argument("--compute-type", default=None, help="compute type for faster_whisper/whisperx (float16/float32/int8)")
    args = parser.parse_args()

    # Choose compute_type: force float32 on CPU to avoid fp16 issues
    compute_type = args.compute_type
    if compute_type is None:
        compute_type = "float32" if args.device == "cpu" else "float16"

    log("="*80)
    log(f"[main] Starting transcription. Args: input={args.input} model={args.model} device={args.device} compute_type={compute_type}")
    try:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        output_dir = input_path.parent
        base = input_path.stem

        log(f"[main] input file info: {safe_path_info(input_path)}")
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "audio.wav"
            log(f"[main] temp audio will be: {audio_path}")

            # extract audio
            extract_audio(str(input_path), str(audio_path))
            log(f"[main] audio file info: {safe_path_info(audio_path)}")

            # transcribe & align
            aligned = transcribe_and_align(str(audio_path), model_name=args.model, device=args.device, compute_type=compute_type)

            # write outputs next to input file
            srt_file = output_dir / f"{base}.srt"
            vtt_file = output_dir / f"{base}.vtt"

            log(f"[main] will write srt to: {srt_file}")
            write_srt(aligned["segments"], str(srt_file))
            write_vtt(aligned["segments"], str(vtt_file))

            log(f"[main] final files: {safe_path_info(srt_file)} ; {safe_path_info(vtt_file)}")
            print(f"✅ Done: {srt_file} {vtt_file}")

    except Exception:
        log("[main] Exception occurred:")
        log(traceback.format_exc())
        print("[ERROR] transcription failed; see transcribe.log for details", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
