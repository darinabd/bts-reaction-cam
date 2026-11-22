import unittest
from types import SimpleNamespace

from recognition import ReactionStabilizer, classify


def category(name, score):
    return SimpleNamespace(category_name=name, score=score)


def face(**values):
    return SimpleNamespace(face_blendshapes=[[category(k, v) for k, v in values.items()]])


def hand(name, score=0.9):
    return SimpleNamespace(gestures=[[category(name, score)]])


class RecognitionTests(unittest.TestCase):
    def test_smile(self):
        self.assertEqual(classify(face(mouthSmileLeft=.75, mouthSmileRight=.8)), "smile")

    def test_surprise(self):
        self.assertEqual(classify(face(jawOpen=.7, browInnerUp=.4)), "surprise")

    def test_wink(self):
        self.assertEqual(classify(face(eyeBlinkLeft=.8, eyeBlinkRight=.06)), "wink")

    def test_gesture_priority(self):
        self.assertEqual(classify(face(mouthSmileLeft=.9, mouthSmileRight=.9),
                                  hand("Victory")), "victory")

    def test_unknown_or_uncertain_gesture(self):
        self.assertIsNone(classify(gesture_result=hand("Victory", .3)))
        self.assertIsNone(classify(gesture_result=hand("Closed_Fist")))

    def test_three_frames_and_minimum_clip_duration(self):
        s = ReactionStabilizer(consecutive_frames=3, min_clip_seconds=2)
        self.assertIsNone(s.update("smile", 0.0))
        self.assertIsNone(s.update("smile", 0.1))
        self.assertEqual(s.update("smile", 0.2), "smile")
        for t in (0.3, 0.4, 0.5):
            self.assertEqual(s.update("victory", t), "smile")
        self.assertEqual(s.update("victory", 2.3), "victory")

    def test_idle_clear(self):
        s = ReactionStabilizer(consecutive_frames=1, min_clip_seconds=1,
                               idle_seconds=.5)
        self.assertEqual(s.update("smile", 0), "smile")
        self.assertEqual(s.update(None, .4), "smile")
        self.assertIsNone(s.update(None, 1.1))


if __name__ == "__main__":
    unittest.main()

