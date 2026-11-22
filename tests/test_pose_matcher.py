import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pose_matcher import (
    FACE_KEYS, FEATURE_LENGTH, PersonalMatcher, distance, signature,
)


def face(**scores):
    return SimpleNamespace(face_blendshapes=[[
        SimpleNamespace(category_name=name, score=value)
        for name, value in scores.items()
    ]])


class PersonalPoseTests(unittest.TestCase):
    def test_feature_size_with_no_landmarks(self):
        empty = SimpleNamespace(face_blendshapes=[], hand_landmarks=[],
                                pose_landmarks=[])
        self.assertEqual(FEATURE_LENGTH, len(signature(empty, empty, empty)))

    def test_expressions_change_feature_vector(self):
        blank = SimpleNamespace(hand_landmarks=[], pose_landmarks=[])
        smiling = signature(face(mouthSmileLeft=.9, mouthSmileRight=.8), blank, blank)
        surprised = signature(face(jawOpen=.9), blank, blank)
        self.assertGreater(distance(smiling, surprised), .05)
        self.assertEqual(smiling[FACE_KEYS.index("mouthSmileLeft")], .9)

    def test_personal_training_predicts_specific_photo(self):
        with tempfile.TemporaryDirectory() as temp:
            matcher = PersonalMatcher(Path(temp) / "calibration.json")
            neutral = [0.] * FEATURE_LENGTH
            smile = neutral.copy()
            smile[0] = .9
            surprise = neutral.copy()
            surprise[2] = .9
            matcher.learn("__neutral__", [neutral] * 4)
            matcher.learn("jimin1.JPG", [smile] * 4)
            matcher.learn("jk1.JPG", [surprise] * 4)
            self.assertEqual(matcher.predict(smile)[0], "jimin1.JPG")
            self.assertEqual(matcher.predict(surprise)[0], "jk1.JPG")
            self.assertIsNone(matcher.predict(neutral)[0])
            restored = PersonalMatcher(matcher.path)
            self.assertEqual(len(restored.samples), 3)

    def test_identical_templates_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as temp:
            matcher = PersonalMatcher(Path(temp) / "calibration.json")
            sample = [0.] * FEATURE_LENGTH
            sample[0] = 1.
            matcher.learn("__neutral__", [[0.] * FEATURE_LENGTH])
            matcher.learn("jimin1.JPG", [sample])
            matcher.learn("jk1.JPG", [sample])
            self.assertIsNone(matcher.predict(sample)[0])


if __name__ == "__main__":
    unittest.main()
