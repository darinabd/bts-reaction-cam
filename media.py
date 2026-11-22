"""Read a user-provided BTS clip (MP4, GIF or still image) on demand."""

from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageSequence
from catalog import ReactionCatalog


EXTENSIONS = (".mp4", ".mov", ".gif", ".png", ".jpg", ".jpeg")


class MemePlayer:
    def __init__(self, assets_dir: Path):
        self.assets_dir = assets_dir
        self.catalog = ReactionCatalog(assets_dir)
        self.name = None
        self.source_name = None
        self.capture = None
        self.image = None
        self.gif_frames = []
        self.gif_durations = []
        self.frame_index = 0
        self.next_frame_at = 0.0
        self.next_video_at = 0.0
        self.last_video_frame = None

    def close(self):
        if self.capture is not None:
            self.capture.release()
        self.capture = None

    def select(self, name: str | None):
        if name == self.name:
            return
        self.close()
        self.name = name
        self.source_name = None
        self.image = None
        self.gif_frames = []
        self.gif_durations = []
        self.last_video_frame = None
        self.frame_index = 0
        self.next_frame_at = 0.0
        self.next_video_at = 0.0

        if name is None:
            return
        asset = self.catalog.choose(name) or next(
            (self.assets_dir / (name + ext) for ext in EXTENSIONS
             if (self.assets_dir / (name + ext)).is_file()),
            None,
        )
        if asset is None:
            return
        self.source_name = asset.name

        if asset.suffix.lower() in (".mp4", ".mov"):
            cap = cv2.VideoCapture(str(asset))
            if cap.isOpened():
                self.capture = cap
            else:
                cap.release()
        elif asset.suffix.lower() == ".gif":
            with Image.open(asset) as gif:
                for frame in ImageSequence.Iterator(gif):
                    self.gif_frames.append(
                        cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
                    )
                    self.gif_durations.append(max(0.04, frame.info.get("duration", 100) / 1000))
        else:
            self.image = cv2.imread(str(asset))

    def frame(self):
        if self.image is not None:
            return self.image

        if self.gif_frames:
            now = time.monotonic()
            if self.next_frame_at == 0:
                self.next_frame_at = now + self.gif_durations[0]
            elif now >= self.next_frame_at:
                self.frame_index = (self.frame_index + 1) % len(self.gif_frames)
                self.next_frame_at = now + self.gif_durations[self.frame_index]
            return self.gif_frames[self.frame_index]

        if self.capture is not None:
            now = time.monotonic()
            if now < self.next_video_at and self.last_video_frame is not None:
                return self.last_video_frame
            okay, frame = self.capture.read()
            if not okay:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                okay, frame = self.capture.read()
            if okay:
                fps = self.capture.get(cv2.CAP_PROP_FPS)
                self.next_video_at = now + 1 / (fps if 1 <= fps <= 120 else 24)
                self.last_video_frame = frame
            return self.last_video_frame
        return None

