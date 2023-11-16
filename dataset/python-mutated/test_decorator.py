import unittest
from patterns.structural.decorator import BoldWrapper, ItalicWrapper, TextTag

class TestTextWrapping(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.raw_string = TextTag('raw but not cruel')

    def test_italic(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(ItalicWrapper(self.raw_string).render(), '<i>raw but not cruel</i>')

    def test_bold(self):
        if False:
            return 10
        self.assertEqual(BoldWrapper(self.raw_string).render(), '<b>raw but not cruel</b>')

    def test_mixed_bold_and_italic(self):
        if False:
            print('Hello World!')
        self.assertEqual(BoldWrapper(ItalicWrapper(self.raw_string)).render(), '<b><i>raw but not cruel</i></b>')