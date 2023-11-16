"""
Generator module tests
"""
import unittest
from txtai.pipeline import Generator

class TestGenerator(unittest.TestCase):
    """
    Sequences tests.
    """

    def testGeneration(self):
        if False:
            while True:
                i = 10
        '\n        Test text pipeline generation\n        '
        model = Generator('hf-internal-testing/tiny-random-gpt2')
        start = 'Hello, how are'
        self.assertIsNotNone(model(start))