"""Turn MediaPipe outputs into stable reaction names.

No machine-learning dependencies are imported here, so the logic can be tested
without a camera or installed models.
"""

from __future__ import annotations

from dataclasses import dataclass


REACTION_LABELS = {
    "smile": "HAPPY / SMILE",
    "surprise": "SHOCKED / SURPRISE",
    "wink": "WINK",
    "thumbs_up": "THUMBS UP",
    "victory": "PEACE SIGN",
    "open_palm": "HELLO / WAVE",
    "love": "I LOVE YOU",
}

GESTURE_MAP = {
    "Thumb_Up": "thumbs_up",
    "Victory": "victory",
    "Open_Palm": "open_palm",
    "ILoveYou": "love",
}


def classify(face_result=None, gesture_result=None) -> str | None:
    """Gesture wins over expression when both are visible.

    Face coefficients measure shapes, not a person's psychological emotion.
    """
    if gesture_result is not None:
        for hand in getattr(gesture_result, "gestures", []):
            if hand:
                top = hand[0]
                name = GESTURE_MAP.get(top.category_name)
                if name and top.score >= 0.67:
                    return name

    if face_result is None or not getattr(face_result, "face_blendshapes", []):
        return None

    shapes = {
        entry.category_name: entry.score
        for entry in face_result.face_blendshapes[0]
    }
    score = lambda key: shapes.get(key, 0.0)

    blink_left = score("eyeBlinkLeft")
    blink_right = score("eyeBlinkRight")
    if max(blink_left, blink_right) > 0.66 and min(blink_left, blink_right) < 0.28:
        return "wink"

    if score("jawOpen") > 0.48 and (
        score("browInnerUp") > 0.18
        or max(score("eyeWideLeft"), score("eyeWideRight")) > 0.22
    ):
        return "surprise"

    if (score("mouthSmileLeft") + score("mouthSmileRight")) / 2 > 0.43:
        return "smile"

    return None


@dataclass
class ReactionStabilizer:
    """Avoid flickering and rapid accidental changes between reaction clips."""

    consecutive_frames: int = 3
    min_clip_seconds: float = 2.0
    idle_seconds: float = 1.3
    current: str | None = None
    candidate: str | None = None
    count: int = 0
    changed_at: float = 0.0
    last_detected_at: float = 0.0

    def update(self, detected: str | None, now: float) -> str | None:
        if detected is not None:
            self.last_detected_at = now

        if detected == self.candidate and detected is not None:
            self.count += 1
        else:
            self.candidate = detected
            self.count = 1 if detected else 0

        if (
            detected is not None
            and detected != self.current
            and self.count >= self.consecutive_frames
            and (self.current is None or now - self.changed_at >= self.min_clip_seconds)
        ):
            self.current = detected
            self.changed_at = now

        if (
            detected is None
            and self.current is not None
            and now - self.changed_at >= self.min_clip_seconds
            and now - self.last_detected_at >= self.idle_seconds
        ):
            self.current = None

        return self.current

