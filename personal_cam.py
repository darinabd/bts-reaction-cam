"""One independently trained webcam expression or pose for each BTS JPG."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import json
import time

import cv2
import numpy as np

from pose_matcher import PersonalMatcher, signature
from recognition import ReactionStabilizer


ROOT = Path(__file__).resolve().parent
COLLECTION = ROOT / "assets" / "collection"
CALIBRATION = ROOT / "calibration.json"
WINDOW = "BTS Reaction Cam - Personal Poses"
WIDTH, HEIGHT = 1280, 760


def photos():
    return sorted(COLLECTION.glob("*.JPG"))


def put(canvas, string, x, y, scale=.66, color=(240, 220, 250), thickness=2):
    cv2.putText(canvas, string, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def fit(image, width=580, height=510):
    if image is None:
        return np.full((height, width, 3), (63, 34, 66), dtype=np.uint8)
    ih, iw = image.shape[:2]
    factor = min(width / iw, height / ih)
    sized = cv2.resize(image, (max(1, int(iw * factor)),
                               max(1, int(ih * factor))))
    result = np.full((height, width, 3), (48, 29, 55), dtype=np.uint8)
    h, w = sized.shape[:2]
    result[(height-h)//2:(height-h)//2+h,
           (width-w)//2:(width-w)//2+w] = sized
    return result


def dashboard(camera, meme, title, footer, small=""):
    screen = np.full((HEIGHT, WIDTH, 3), (29, 17, 29), dtype=np.uint8)
    cv2.rectangle(screen, (0, 0), (WIDTH, 8), (218, 128, 226), -1)
    put(screen, "BTS REACTION CAM", 36, 61, 1.03, thickness=3)
    put(screen, title[:70], 36, 101, .57)
    put(screen, "YOUR CAMERA", 42, 145, .55, (218, 128, 226))
    put(screen, "YOUR BTS MEME", 665, 145, .55, (218, 128, 226))
    screen[155:665, 35:615] = fit(camera)
    screen[155:665, 660:1240] = fit(meme)
    put(screen, footer[:84], 36, 707, .56)
    if small:
        put(screen, small[:95], 36, 738, .43, (194, 161, 205), 1)
    return screen


def webcam(index):
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        raise RuntimeError("Camera is unavailable. Check macOS camera permissions.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return camera


def detectors(stack):
    import mediapipe as mp

    model_files = {
        "face": ROOT / "models" / "face_landmarker.task",
        "hands": ROOT / "models" / "gesture_recognizer.task",
        "pose": ROOT / "models" / "pose_landmarker_lite.task",
    }
    if not all(file.is_file() for file in model_files.values()):
        raise FileNotFoundError("Missing models: run `python setup_models.py` first.")
    mode = mp.tasks.vision.RunningMode.VIDEO
    face = stack.enter_context(mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_files["face"])),
            running_mode=mode, num_faces=1, output_face_blendshapes=True)))
    hands = stack.enter_context(mp.tasks.vision.GestureRecognizer.create_from_options(
        mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_files["hands"])),
            running_mode=mode, num_hands=2)))
    pose = stack.enter_context(mp.tasks.vision.PoseLandmarker.create_from_options(
        mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_files["pose"])),
            running_mode=mode, num_poses=1)))
    return mp, face, hands, pose


def extract(frame, timestamp, models):
    mp, face, hands, pose = models
    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return signature(
        face.detect_for_video(mp_image, timestamp),
        hands.recognize_for_video(mp_image, timestamp),
        pose.detect_for_video(mp_image, timestamp),
    )


def train(camera_index=0):
    all_photos = photos()
    if not all_photos:
        raise FileNotFoundError("No JPG memes inside assets/collection")
    matcher = PersonalMatcher(CALIBRATION)
    hints = json.loads((ROOT / "pose_guides.json").read_text(encoding="utf-8"))
    items = [("__neutral__", None)] + [(p.name, p) for p in all_photos]
    # Resume at first untrained pose when training was interrupted.
    index = next((i for i, (name, _) in enumerate(items)
                  if name not in matcher.samples), 0)
    camera = webcam(camera_index)
    start = time.monotonic()
    last_timestamp = -1
    collecting: list[list[float]] | None = None
    last_capture_at = 0.
    try:
        with ExitStack() as stack:
            models = detectors(stack)
            while True:
                okay, camera_frame = camera.read()
                if not okay:
                    break
                camera_frame = cv2.flip(camera_frame, 1)
                now = time.monotonic()
                timestamp = max(last_timestamp + 1, int((now - start) * 1000))
                last_timestamp = timestamp
                features = extract(camera_frame, timestamp, models)
                label, path = items[index]
                photo = cv2.imread(str(path)) if path is not None else None
                if collecting is not None and now - last_capture_at >= .08:
                    collecting.append(features)
                    last_capture_at = now
                    if len(collecting) >= 12:
                        matcher.learn(label, collecting)
                        print("Saved pose for", label)
                        collecting = None
                        if index < len(items) - 1:
                            index += 1
                progress = sum(name in matcher.samples for name, _ in items)
                if label == "__neutral__":
                    headline = "STEP 0: LOOK AT CAMERA RELAXED (NO GESTURE)"
                else:
                    headline = f"PHOTO {index}/29: {label}  |  {hints.get(label, 'Your unique pose')}"
                footer = (f"CAPTURING: {len(collecting)}/12 - HOLD STILL!"
                          if collecting is not None else
                          "SPACE: learn pose   N: next   B: previous   Q: finish")
                extra = f"Trained: {progress}/30  |  Captures save automatically"
                cv2.imshow(WINDOW, dashboard(camera_frame, photo, headline, footer, extra))
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if collecting is None:
                    if key == ord(" "):
                        collecting = []
                        last_capture_at = 0.
                    elif key in (ord("n"), ord("N")):
                        index = min(index + 1, len(items) - 1)
                    elif key in (ord("b"), ord("B")):
                        index = max(index - 1, 0)
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print(f"Calibrated {len(matcher.samples)} / {len(items)} poses.")


def run(camera_index=0):
    matcher = PersonalMatcher(CALIBRATION)
    all_photos = photos()
    missing = [p.name for p in all_photos if p.name not in matcher.samples]
    if "__neutral__" not in matcher.samples or missing:
        raise RuntimeError(
            f"Train all poses first with `python app.py --train` "
            f"({len(missing)} JPGs not yet trained)."
        )
    camera = webcam(camera_index)
    stabilizer = ReactionStabilizer(consecutive_frames=4,
                                    min_clip_seconds=2.0, idle_seconds=.7)
    last_selected = None
    active_photo = None
    start = time.monotonic()
    last_timestamp = -1
    try:
        with ExitStack() as stack:
            models = detectors(stack)
            while True:
                okay, camera_frame = camera.read()
                if not okay:
                    break
                camera_frame = cv2.flip(camera_frame, 1)
                now = time.monotonic()
                timestamp = max(last_timestamp + 1, int((now - start) * 1000))
                last_timestamp = timestamp
                observed = extract(camera_frame, timestamp, models)
                detected, score = matcher.predict(observed)
                selected = stabilizer.update(detected, now)
                if selected != last_selected:
                    active_photo = (cv2.imread(str(COLLECTION / selected))
                                    if selected else None)
                    last_selected = selected
                headline = "MATCH YOUR TRAINED EXPRESSION OR POSE"
                footer = (f"MATCH: {selected}" if selected else
                          "NO CONFIDENT MATCH - try a clearer / stronger pose")
                small = "Q / ESC: quit   |   Retrain any photo: python app.py --train"
                cv2.imshow(WINDOW, dashboard(
                    camera_frame, active_photo, headline, footer, small))
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()
