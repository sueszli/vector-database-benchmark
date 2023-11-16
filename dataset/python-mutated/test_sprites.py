import unittest
from asciimatics.paths import Path
from asciimatics.sprites import Sam, Arrow, Plot

class TestSprites(unittest.TestCase):

    def test_init(self):
        if False:
            while True:
                i = 10
        self.assertIsNotNone(Sam(None, Path()))
        self.assertIsNotNone(Arrow(None, Path()))
        self.assertIsNotNone(Plot(None, Path()))
if __name__ == '__main__':
    unittest.main()