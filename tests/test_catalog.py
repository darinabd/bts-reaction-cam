import unittest
from pathlib import Path

from catalog import ReactionCatalog
from recognition import REACTION_LABELS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReactionCatalogTests(unittest.TestCase):
    def test_all_reactions_have_images(self):
        catalog = ReactionCatalog(PROJECT_ROOT / "assets")
        self.assertEqual(set(catalog.mapping), set(REACTION_LABELS))
        for reaction in REACTION_LABELS:
            with self.subTest(reaction=reaction):
                self.assertGreater(len(catalog.options(reaction)), 0)

    def test_all_29_uploaded_memes_are_used(self):
        catalog = ReactionCatalog(PROJECT_ROOT / "assets")
        used = {path.name for reaction in catalog.mapping for path in catalog.options(reaction)}
        uploaded = {p.name for p in (PROJECT_ROOT / "assets" / "collection").glob("*.JPG")}
        self.assertEqual(29, len(uploaded))
        self.assertEqual(used, uploaded)

    def test_random_choice_avoids_immediate_repeats(self):
        catalog = ReactionCatalog(PROJECT_ROOT / "assets")
        first = catalog.choose("surprise")
        second = catalog.choose("surprise")
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
