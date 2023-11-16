"""
Tests for L{twisted.python.textattributes}.
"""
from twisted.python._textattributes import DefaultFormattingState
from twisted.trial import unittest

class DefaultFormattingStateTests(unittest.TestCase):
    """
    Tests for L{twisted.python._textattributes.DefaultFormattingState}.
    """

    def test_equality(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{DefaultFormattingState}s are always equal to other\n        L{DefaultFormattingState}s.\n        '
        self.assertEqual(DefaultFormattingState(), DefaultFormattingState())
        self.assertNotEqual(DefaultFormattingState(), 'hello')