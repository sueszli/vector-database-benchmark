"""
Sequences module tests
"""
import unittest
from txtai.pipeline import Sequences

class TestSequences(unittest.TestCase):
    """
    Sequences tests.
    """

    def testGeneration(self):
        if False:
            print('Hello World!')
        '\n        Test text2text pipeline generation\n        '
        model = Sequences('t5-small')
        self.assertEqual(model('Testing the model', prefix='translate English to German: '), 'Das Modell zu testen')