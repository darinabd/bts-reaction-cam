import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GuideTests(unittest.TestCase):
    def test_each_photo_has_its_own_guide(self):
        guides = json.loads((ROOT / "pose_guides.json").read_text(encoding="utf-8"))
        photos = {p.name for p in (ROOT / "assets" / "collection").glob("*.JPG")}
        self.assertEqual(len(photos), 29)
        self.assertEqual(set(guides), photos)
        self.assertTrue(all(guides.values()))


if __name__ == "__main__":
    unittest.main()
