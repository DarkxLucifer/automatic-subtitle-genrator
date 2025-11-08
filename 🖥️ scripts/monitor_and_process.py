import time
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

WATCH_DIR = Path("examples").resolve()
SUPPORTED_EXTS = {".mp4", ".mkv", ".wav", ".mp3", ".mov"}

class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() in SUPPORTED_EXTS:
            print(f"[+] New file detected: {path.name}")
            cmd = [
                "python", "-m", "src.autosub.transcribe_accurate",
                "--input", str(path),
                "--model", "medium",
                "--device", "cuda"
            ]
            print("Running:", " ".join(cmd))
            subprocess.Popen(cmd)

def main():
    WATCH_DIR.mkdir(exist_ok=True)
    observer = Observer()
    event_handler = FileHandler()
    observer.schedule(event_handler, str(WATCH_DIR), recursive=False)
    observer.start()
    print(f"👀 Watching {WATCH_DIR} for new media files...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
