# src/autosub/transcribe_accurate.py
import argparse
import os
import subprocess
import tempfile
import traceback
from pathlib import Path
import sys
import torch

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

def transcribe_and_align(audio_path: str, model_name: str = "medium", device: str = "cuda"):
    log(f"[transcribe_and_align] model={model_name}, device={device}, audio={audio_path}")
    model = whisperx.load_model(model_name, device=device)
    log("[transcribe_and_align] whisper model loaded")
    result = model.transcribe(audio_path)
    log("[transcribe_and_align] initial transcript done; loading align model (cpu)")
    align_model, metadata = whisperx.load_align_model(result["language"], device="cpu")
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
    log(f"[write_srt] wrote {path} ({os.path.getsize(path)} bytes)")

def write_vtt(segments, path: str):
    log(f"[write_vtt] writing vtt to: {path}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start = seconds_to_srt_timestamp(seg["start"]).replace(",", ".")
            end = seconds_to_srt_timestamp(seg["end"]).replace(",", ".")
            text = seg.get("text", "").strip()
            f.write(f"{start} --> {end}\n{text}\n\n")
    log(f"[write_vtt] wrote {path} ({os.path.getsize(path)} bytes)")

def safe_path_info(p: Path):
    try:
        return f"{p} exists={p.exists()} size={p.stat().st_size if p.exists() else 'n/a'}"
    except Exception as e:
        return f"{p} exists=ERROR ({e})"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--model", default="medium")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    log("="*80)
    log(f"[main] Starting transcription. Args: {args}")
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
            aligned = transcribe_and_align(str(audio_path), model_name=args.model, device=args.device)

            # write outputs next to input file
            srt_file = output_dir / f"{base}.srt"
            vtt_file = output_dir / f"{base}.vtt"

            log(f"[main] will write srt to: {srt_file}")
            write_srt(aligned["segments"], str(srt_file))
            write_vtt(aligned["segments"], str(vtt_file))

            log(f"[main] final files: {safe_path_info(srt_file)} ; {safe_path_info(vtt_file)}")
            print(f"✅ Done: {srt_file} {vtt_file}")

    except Exception as e:
        log("[main] Exception occurred:")
        log(traceback.format_exc())
        print("[ERROR] transcription failed; see transcribe.log for details", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
