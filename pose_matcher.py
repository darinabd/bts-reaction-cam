"""Small, local, user-trained expression + gesture + pose matcher.

Each JPG gets its own webcam reference. The classifier does not try to infer
what happens in a BTS photo; the user records the intended action once.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


FACE_KEYS = (
    "jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthPucker", "mouthFunnel", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthRollLower", "mouthRollUpper",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeWideLeft", "eyeWideRight",
    "eyeSquintLeft", "eyeSquintRight", "browInnerUp", "browDownLeft",
    "browDownRight", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight", "noseSneerLeft", "noseSneerRight",
)
POSE_JOINTS = (0, 11, 12, 13, 14, 15, 16, 23, 24)
HAND_TIPS = (4, 8, 12, 16, 20)
FEATURE_LENGTH = len(FACE_KEYS) + 1 + 1 + len(POSE_JOINTS) * 2 + 2 * 14


def signature(face_result, gesture_result, pose_result) -> list[float]:
    """Produce a consistent vector from MediaPipe results for video frames."""
    face_shapes = getattr(face_result, "face_blendshapes", [])
    face_present = bool(face_shapes)
    scores = ({item.category_name: item.score for item in face_shapes[0]}
              if face_present else {})
    values = [float(scores.get(name, 0.0)) for name in FACE_KEYS]
    values.append(float(face_present))

    poses = getattr(pose_result, "pose_landmarks", [])
    pose = poses[0] if poses else None
    pose_present = pose is not None and len(pose) > 24
    values.append(float(pose_present))
    if pose_present:
        left, right = pose[11], pose[12]
        center_x = (left.x + right.x) / 2
        center_y = (left.y + right.y) / 2
        scale = max(math.hypot(left.x - right.x, left.y - right.y), 0.13)
        for index in POSE_JOINTS:
            point = pose[index]
            visible = getattr(point, "visibility", 1.0) > 0.45
            if visible:
                values.extend((
                    max(-3., min(3., (point.x - center_x) / scale)) * 0.25,
                    max(-3., min(3., (point.y - center_y) / scale)) * 0.25,
                ))
            else:
                values.extend((0., 0.))
    else:
        values.extend([0.] * (len(POSE_JOINTS) * 2))

    # Wrist position relative to shoulders + relative shape of fingers.
    # Sorting by screen x gives a stable two-hand order in a mirrored webcam.
    hands = list(getattr(gesture_result, "hand_landmarks", []) or [])
    hands.sort(key=lambda points: points[0].x)
    for landmarks in hands[:2]:
        wrist = landmarks[0]
        palm_size = max(math.hypot(landmarks[5].x - landmarks[17].x,
                                   landmarks[5].y - landmarks[17].y), 0.05)
        if pose_present:
            x = (wrist.x - center_x) / scale
            y = (wrist.y - center_y) / scale
        else:
            x = (wrist.x - .5) * 3
            y = (wrist.y - .5) * 3
        vector = [1., max(-3., min(3., x)) * .25,
                  max(-3., min(3., y)) * .25, 0.]
        for tip in HAND_TIPS:
            point = landmarks[tip]
            vector.extend((
                max(-2., min(2., (point.x - wrist.x) / palm_size)) * .25,
                max(-2., min(2., (point.y - wrist.y) / palm_size)) * .25,
            ))
        values.extend(vector)
    for _ in range(2 - min(2, len(hands))):
        values.extend([0.] * 14)

    if len(values) != FEATURE_LENGTH:
        raise ValueError(f"Invalid feature size {len(values)}, expected {FEATURE_LENGTH}")
    return values


def distance(left: list[float], right: list[float]) -> float:
    if len(left) != FEATURE_LENGTH or len(right) != FEATURE_LENGTH:
        raise ValueError("Incompatible calibration data; retrain poses")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)) / len(left))


class PersonalMatcher:
    def __init__(self, path: Path):
        self.path = path
        self.samples: dict[str, list[float]] = {}
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != 1:
                raise ValueError("Unsupported calibration version")
            self.samples = payload.get("samples", {})

    def save(self):
        payload = {"schema_version": 1, "samples": self.samples}
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp.replace(self.path)

    def learn(self, filename: str, frames: list[list[float]]):
        if not frames:
            raise ValueError("No pose samples captured")
        if any(len(row) != FEATURE_LENGTH for row in frames):
            raise ValueError("Invalid pose sample")
        self.samples[filename] = [
            sum(row[i] for row in frames) / len(frames)
            for i in range(FEATURE_LENGTH)
        ]
        self.save()

    def predict(self, observed: list[float]) -> tuple[str | None, float]:
        if "__neutral__" not in self.samples:
            return None, 0.
        scores = sorted(
            (distance(observed, ref), name)
            for name, ref in self.samples.items()
        )
        closest_dist, closest_name = scores[0]
        if closest_name == "__neutral__":
            return None, closest_dist
        neutral_dist = distance(observed, self.samples["__neutral__"])
        if closest_dist > 0.18 or closest_dist >= neutral_dist * .78:
            return None, closest_dist
        # Similar templates are ambiguous; suppress them rather than guessing.
        alternatives = [score for score, name in scores[1:] if name != "__neutral__"]
        if alternatives and alternatives[0] - closest_dist < 0.012:
            return None, closest_dist
        return closest_name, closest_dist
