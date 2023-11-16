import unittest
from instapy.util import evaluate_mandatory_words

class UtilsTests(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_evaluate_mandatory_words(self):
        if False:
            return 10
        text = 'a b c d e f g'
        self.assertTrue(evaluate_mandatory_words(text, ['a', 'B', 'c']))
        self.assertTrue(evaluate_mandatory_words(text, ['a', 'B', 'z']))
        self.assertTrue(evaluate_mandatory_words(text, ['x', 'B', 'z']))
        self.assertFalse(evaluate_mandatory_words(text, ['x', 'y', 'z']))
        self.assertTrue(evaluate_mandatory_words(text, [['a', 'f', 'e']]))
        self.assertTrue(evaluate_mandatory_words(text, ['a', ['x', 'y', 'z']]))
        self.assertTrue(evaluate_mandatory_words(text, [['x', 'y', 'z'], 'a']))
        self.assertFalse(evaluate_mandatory_words(text, [['x', 'y', 'z'], 'v']))
        self.assertTrue(evaluate_mandatory_words(text, [['a', 'b', 'd'], 'v']))
        self.assertTrue(evaluate_mandatory_words(text, [['a', 'b', ['d', 'x']], 'v']))
        self.assertFalse(evaluate_mandatory_words(text, [['a', 'z', ['d', 'x']], 'v']))
        self.assertTrue(evaluate_mandatory_words(text, [['a', 'b', [['d', 'e'], 'x']]]))
        self.assertFalse(evaluate_mandatory_words(text, [['a', 'b', [['d', 'z'], 'x']]]))