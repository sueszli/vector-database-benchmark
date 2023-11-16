"""
Tokenizer module tests
"""
import unittest
from txtai.pipeline import Tokenizer

class TestTokenizer(unittest.TestCase):
    """
    Tokenizer tests.
    """

    def testAlphanumTokenize(self):
        if False:
            return 10
        '\n        Test alphanumeric tokenization\n        '
        self.assertEqual(Tokenizer.tokenize('Y this is a test!'), ['test'])
        self.assertEqual(Tokenizer.tokenize('abc123 ABC 123'), ['abc123', 'abc'])

    def testStandardTokenize(self):
        if False:
            print('Hello World!')
        '\n        Test standard tokenization\n        '
        tokenizer = Tokenizer()
        tests = [('Y this is a test!', ['y', 'this', 'is', 'a', 'test']), ('abc123 ABC 123', ['abc123', 'abc', '123']), ('Testing hy-phenated words', ['testing', 'hy', 'phenated', 'words']), ('111-111-1111', ['111', '111', '1111']), ('Test.1234', ['test', '1234'])]
        for (test, result) in tests:
            self.assertEqual(tokenizer(test), result)