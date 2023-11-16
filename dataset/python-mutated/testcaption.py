"""
Caption module tests
"""
import unittest
from PIL import Image
from txtai.pipeline import Caption
from utils import Utils

class TestCaption(unittest.TestCase):
    """
    Caption tests.
    """

    def testCaption(self):
        if False:
            i = 10
            return i + 15
        '\n        Test captions\n        '
        caption = Caption()
        self.assertEqual(caption(Image.open(Utils.PATH + '/books.jpg')), 'a book shelf filled with books and a stack of books')