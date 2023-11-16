import unittest
from qtconsole.styles import dark_color, dark_style

class TestStyles(unittest.TestCase):

    def test_dark_color(self):
        if False:
            while True:
                i = 10
        self.assertTrue(dark_color('#000000'))
        self.assertTrue(not dark_color('#ffff66'))
        self.assertTrue(dark_color('#80807f'))
        self.assertTrue(not dark_color('#808080'))

    def test_dark_style(self):
        if False:
            return 10
        self.assertTrue(dark_style('monokai'))
        self.assertTrue(not dark_style('default'))