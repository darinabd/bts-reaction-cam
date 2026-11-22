"""BTS Reaction Cam: personalized 1-to-1 pose/expression per meme.

Run `python app.py --train` to record the 29 triggers, then `python app.py`.
`--demo` retains the original simple 7-reaction keyboard test.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
import time

import cv2
import numpy as np

from media import MemePlayer
from recognition import REACTION_LABELS, ReactionStabilizer, classify


ROOT = Path(__file__).resolve().parent
SIZE = (1280, 760)
PANELS = ((36, 142, 585, 500), (659, 142, 585, 500))
PALETTE = {
    "background": (30, 15, 24),
    "card": (52, 33, 48),
    "accent": (231, 147, 217),
    "soft": (200, 177, 201),
    "white": (251, 244, 252),
}


def draw_text(canvas, text: str, x: int, y: int, scale=0.65, color=None,
              thickness=2):
    cv2.putText(
        canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
        color or PALETTE["white"], thickness, cv2.LINE_AA,
    )


def rounded_card(canvas, x, y, w, h, color, radius=22):
    cv2.rectangle(canvas, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(canvas, (x, y + radius), (x + w, y + h - radius), color, -1)
    for center in ((x + radius, y + radius), (x + w - radius, y + radius),
                   (x + radius, y + h - radius), (x + w - radius, y + h - radius)):
        cv2.circle(canvas, center, radius, color, -1, cv2.LINE_AA)


def cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Fill an area without stretching faces or squeezing video."""
    ih, iw = image.shape[:2]
    if ih <= 0 or iw <= 0:
        raise ValueError("Empty input image")
    factor = max(width / iw, height / ih)
    nw, nh = max(width, round(iw * factor)), max(height, round(ih * factor))
    scaled = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    sx, sy = (nw - width) // 2, (nh - height) // 2
    return scaled[sy:sy + height, sx:sx + width]


def placeholder(width: int, height: int, name: str | None) -> np.ndarray:
    canvas = np.full((height, width, 3), (63, 39, 70), dtype=np.uint8)
    cv2.circle(canvas, (width // 2, height // 2 - 15), 150, (98, 56, 102), -1,
               cv2.LINE_AA)
    cv2.circle(canvas, (width // 2, height // 2 - 15), 120, (145, 79, 145), 5,
               cv2.LINE_AA)
    if name is None:
        draw_text(canvas, "WAITING FOR A REACTION", 78, 230, 0.72)
        draw_text(canvas, "Smile or show a hand gesture", 87, 275, 0.57)
    else:
        label = REACTION_LABELS[name]
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0][0]
        draw_text(canvas, label, (width - text_width) // 2, 220, 0.85)
        draw_text(canvas, "Add assets/" + name + ".mp4", 104, 265, 0.58)
        draw_text(canvas, "or .gif / .png / .jpg", 132, 294, 0.58)
    return canvas


def make_dashboard(camera_frame, meme_frame, reaction, detection, demo, source_name=None):
    canvas = np.full((SIZE[1], SIZE[0], 3), PALETTE["background"], dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (SIZE[0], 8), PALETTE["accent"], -1)
    draw_text(canvas, "BTS REACTION CAM", 36, 62, 1.05, thickness=3)
    draw_text(canvas, "LIVE CAMERA + REACTION MEMES", 38, 94, 0.52,
              PALETTE["soft"], 1)
    draw_text(canvas, "YOU", PANELS[0][0], 131, 0.58, PALETTE["accent"])
    draw_text(canvas, "YOUR BTS MEME", PANELS[1][0], 131, 0.58,
              PALETTE["accent"])

    for x, y, w, h in PANELS:
        rounded_card(canvas, x - 3, y - 3, w + 6, h + 6, PALETTE["card"])

    x, y, w, h = PANELS[0]
    canvas[y:y+h, x:x+w] = cover(camera_frame, w, h)
    x, y, w, h = PANELS[1]
    canvas[y:y+h, x:x+w] = cover(
        meme_frame if meme_frame is not None else placeholder(w, h, reaction), w, h
    )

    label = REACTION_LABELS[reaction] if reaction else "WAITING..."
    draw_text(canvas, "NOW PLAYING: " + label, 38, 688, 0.65)
    if source_name:
        draw_text(canvas, "MEME: " + source_name, 38, 710, 0.45,
                  PALETTE["soft"], 1)
    draw_text(canvas, ("DEMO MODE" if demo else "DETECTED: " +
                       (REACTION_LABELS[detection] if detection else "NONE")),
              662, 688, 0.52, PALETTE["soft"], 1)
    draw_text(canvas, "1-7: test reactions   |   0: clear   |   Q / ESC: quit",
              38, 731, 0.5, PALETTE["soft"], 1)
    return canvas


def build_detectors(stack: ExitStack):
    import mediapipe as mp

    face_path = ROOT / "models" / "face_landmarker.task"
    hand_path = ROOT / "models" / "gesture_recognizer.task"
    if not face_path.is_file() or not hand_path.is_file():
        raise FileNotFoundError(
            "Missing MediaPipe models. First run: python setup_models.py"
        )

    mode = mp.tasks.vision.RunningMode.VIDEO
    face = stack.enter_context(mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(face_path)),
            running_mode=mode,
            num_faces=1,
            output_face_blendshapes=True,
        )
    ))
    hands = stack.enter_context(mp.tasks.vision.GestureRecognizer.create_from_options(
        mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_path)),
            running_mode=mode,
            num_hands=2,
        )
    ))
    return mp, face, hands


def run(camera_index: int, demo: bool):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Grant camera access to Terminal/VS Code "
            "in macOS System Settings > Privacy & Security > Camera."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    clip = MemePlayer(ROOT / "assets")
    stabilizer = ReactionStabilizer()
    number_keys = list(REACTION_LABELS)
    start = time.monotonic()
    last_timestamp = -1

    try:
        with ExitStack() as stack:
            mp, face, hands = (None, None, None) if demo else build_detectors(stack)
            while True:
                okay, frame = cap.read()
                if not okay:
                    print("Camera frame unavailable; stopping.")
                    break
                frame = cv2.flip(frame, 1)
                detected = None
                now = time.monotonic()

                if not demo:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb = np.ascontiguousarray(rgb)
                    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp = max(last_timestamp + 1, int((now - start) * 1000))
                    last_timestamp = timestamp
                    detected = classify(
                        face.detect_for_video(mp_frame, timestamp),
                        hands.recognize_for_video(mp_frame, timestamp),
                    )
                    reaction = stabilizer.update(detected, now)
                else:
                    reaction = stabilizer.current

                clip.select(reaction)
                dashboard = make_dashboard(
                    frame, clip.frame(), reaction, detected, demo, clip.source_name
                )
                cv2.imshow("BTS Reaction Cam", dashboard)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key == ord("0"):
                    stabilizer.current = None
                    stabilizer.changed_at = 0
                elif ord("1") <= key <= ord("7"):
                    stabilizer.current = number_keys[key - ord("1")]
                    stabilizer.changed_at = now
                    stabilizer.last_detected_at = now
    finally:
        clip.close()
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Interactive BTS meme reaction cam")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera number (default 0)")
    parser.add_argument("--demo", action="store_true",
                        help="No MediaPipe needed; switch clips using keys 1-7")
    parser.add_argument("--train", action="store_true",
                        help="Teach a separate expression or pose for each JPG")
    args = parser.parse_args()
    if args.demo:
        run(args.camera, True)
    else:
        from personal_cam import run as run_personal, train
        if args.train:
            train(args.camera)
        else:
            run_personal(args.camera)


if __name__ == "__main__":
    main()

