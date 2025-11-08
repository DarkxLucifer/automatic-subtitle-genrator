# AutoSub — Automatic Subtitle Creation 🎬

AutoSub automatically generates **accurate subtitles** (`.srt` and `.vtt`)  
from video/audio files using [WhisperX](https://github.com/m-bain/whisperX).

### ✨ Features
- Supports audio & video (MP4, MKV, MP3, WAV, MOV)
- Uses **WhisperX** for word-aligned, high-accuracy transcription
- Optional speaker diarization (via `pyannote.audio`)
- Automatic file watcher — runs when a file is added to `examples/`
- GitHub Action auto-commits generated subtitles back to the repo

---

## ⚙️ Installation
```bash
git clone https://github.com/<yourname>/autosub.git
cd autosub
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
