"""
ImageHash module tests
"""
import unittest
from PIL import Image
from txtai.pipeline import ImageHash
from utils import Utils

class TestImageHash(unittest.TestCase):
    """
    ImageHash tests.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        '\n        Caches an image to hash\n        '
        cls.image = Image.open(Utils.PATH + '/books.jpg')

    def testArray(self):
        if False:
            while True:
                i = 10
        '\n        Test numpy return type\n        '
        ihash = ImageHash(strings=False)
        self.assertEqual(ihash(self.image).shape, (64,))

    def testAverage(self):
        if False:
            return 10
        '\n        Test average hash\n        '
        ihash = ImageHash('average')
        self.assertIn(ihash(self.image), ['0859dd04bfbfbf00', '0859dd04ffbfbf00'])

    def testColor(self):
        if False:
            i = 10
            return i + 15
        '\n        Test color hash\n        '
        ihash = ImageHash('color')
        self.assertIn(ihash(self.image), ['1ffffe02000e000c0e0000070000', '1ff8fe03000e00070e0000070000'])

    def testDifference(self):
        if False:
            while True:
                i = 10
        '\n        Test difference hash\n        '
        ihash = ImageHash('difference')
        self.assertEqual(ihash(self.image), 'd291996d6969686a')

    def testPerceptual(self):
        if False:
            print('Hello World!')
        '\n        Test perceptual hash\n        '
        ihash = ImageHash('perceptual')
        self.assertEqual(ihash(self.image), '8be8418577b331b9')

    def testWavelet(self):
        if False:
            i = 10
            return i + 15
        '\n        Test wavelet hash\n        '
        ihash = ImageHash('wavelet')
        self.assertEqual(ihash(Utils.PATH + '/books.jpg'), '68015d85bfbf3f00')