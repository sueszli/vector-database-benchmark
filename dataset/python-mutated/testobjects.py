"""
Objects module tests
"""
import unittest
from txtai.pipeline import Objects
from utils import Utils

class TestObjects(unittest.TestCase):
    """
    Object detection tests.
    """

    def testClassification(self):
        if False:
            while True:
                i = 10
        '\n        Test object detection using an image classification model\n        '
        objects = Objects(classification=True, threshold=0.3)
        self.assertEqual(objects(Utils.PATH + '/books.jpg')[0][0], 'library')

    def testDetection(self):
        if False:
            while True:
                i = 10
        '\n        Test object detection using an object detection model\n        '
        objects = Objects()
        self.assertEqual(objects(Utils.PATH + '/books.jpg')[0][0], 'book')

    def testFlatten(self):
        if False:
            i = 10
            return i + 15
        '\n        Test object detection using an object detection model, flatten to return only objects\n        '
        objects = Objects()
        self.assertEqual(objects(Utils.PATH + '/books.jpg', flatten=True)[0], 'book')