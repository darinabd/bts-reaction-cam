"""Select a random user-provided meme, avoiding immediate repeats."""

from __future__ import annotations

import json
from pathlib import Path
import random


SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".gif", ".png", ".jpg", ".jpeg"}


class ReactionCatalog:
    def __init__(self, assets_dir: Path, manifest_path: Path | None = None):
        self.assets_dir = assets_dir
        self.manifest_path = manifest_path or assets_dir.parent / "reactions.json"
        self.mapping = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.last_by_reaction: dict[str, Path] = {}

    def options(self, reaction: str) -> list[Path]:
        files = []
        collection = (self.assets_dir / "collection").resolve()
        for filename in self.mapping.get(reaction, []):
            candidate = (collection / filename).resolve()
            if (
                candidate.is_relative_to(collection)
                and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
                and candidate.is_file()
            ):
                files.append(candidate)
        return files

    def choose(self, reaction: str) -> Path | None:
        options = self.options(reaction)
        if not options:
            return None
        previous = self.last_by_reaction.get(reaction)
        not_last = [option for option in options if option != previous]
        selected = random.choice(not_last or options)
        self.last_by_reaction[reaction] = selected
        return selected
