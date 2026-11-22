"""Download the official Google MediaPipe task models once."""

from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
MODELS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "gesture_recognizer.task": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}


def main():
    folder = ROOT / "models"
    folder.mkdir(parents=True, exist_ok=True)
    for filename, url in MODELS.items():
        dest = folder / filename
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"Already downloaded: {filename}")
            continue
        part = dest.with_suffix(".part")
        print(f"Downloading {filename} from Google...")
        try:
            with urlopen(url, timeout=60) as response, part.open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            if part.stat().st_size < 1_000_000:
                raise ValueError("Model download was unexpectedly small")
            part.replace(dest)
        except Exception:
            part.unlink(missing_ok=True)
            raise
        print(f"Saved: {dest.name} ({dest.stat().st_size // 1024 // 1024} MB)")


if __name__ == "__main__":
    main()

