"""Tests for findbestmatch.py"""
import unittest
import os.path
test_path = os.path.split(__file__)[0]
import sys
sys.path.append('.')
from pywinauto import findbestmatch
from pywinauto.windows import win32structures

class TestFindBestMatch(unittest.TestCase):

    def testclean_text_1(self):
        if False:
            return 10
        'Test for _clean_non_chars (alphanumeric symbols)'
        s = 'nothingremovedhere'
        result = findbestmatch._clean_non_chars(s)
        self.assertEqual(s, result)

    def testclean_text_2(self):
        if False:
            for i in range(10):
                print('nop')
        'Test for _clean_non_chars (special symbols)'
        s = '#$%#^$%&**'
        result = findbestmatch._clean_non_chars(s)
        self.assertEqual('', result)

    def testclean_text_3(self):
        if False:
            i = 10
            return i + 15
        'Test for _clean_non_chars (empty string)'
        s = ''
        result = findbestmatch._clean_non_chars(s)
        self.assertEqual('', result)

class DummyCtrl:

    def __init__(self, l, t, r, b):
        if False:
            return 10
        self.rect = win32structures.RECT(l, t, r, b)

    def rectangle(self):
        if False:
            print('Hello World!')
        return self.rect

class TestIsAboveOrToLeft(unittest.TestCase):

    def testSameRect(self):
        if False:
            i = 10
            return i + 15
        'both rectangles are the same so false'
        other = DummyCtrl(10, 20, 200, 40)
        this = DummyCtrl(10, 20, 200, 40)
        result = findbestmatch.is_above_or_to_left(this, other)
        self.assertEqual(result, False)

    def testToLeft(self):
        if False:
            while True:
                i = 10
        other = DummyCtrl(10, 20, 200, 40)
        this = DummyCtrl(100, 20, 200, 40)
        result = findbestmatch.is_above_or_to_left(this, other)
        self.assertEqual(result, True)

    def testAbove(self):
        if False:
            for i in range(10):
                print('nop')
        other = DummyCtrl(10, 10, 200, 30)
        this = DummyCtrl(10, 20, 200, 40)
        result = findbestmatch.is_above_or_to_left(this, other)
        self.assertEqual(result, True)

    def testLeftAndTop(self):
        if False:
            print('Hello World!')
        other = DummyCtrl(5, 10, 200, 20)
        this = DummyCtrl(10, 20, 200, 40)
        result = findbestmatch.is_above_or_to_left(this, other)
        self.assertEqual(result, True)

    def testBelow(self):
        if False:
            print('Hello World!')
        other = DummyCtrl(10, 120, 200, 140)
        this = DummyCtrl(10, 20, 20, 40)
        result = findbestmatch.is_above_or_to_left(this, other)
        self.assertEqual(result, False)

    def testToRight(self):
        if False:
            i = 10
            return i + 15
        other = DummyCtrl(110, 20, 120, 40)
        this = DummyCtrl(10, 20, 20, 40)
        result = findbestmatch.is_above_or_to_left(this, other)
        self.assertEqual(result, False)

    def testTopLeftInSideControl(self):
        if False:
            for i in range(10):
                print('nop')
        other = DummyCtrl(15, 25, 120, 40)
        this = DummyCtrl(10, 20, 20, 40)
        result = findbestmatch.is_above_or_to_left(this, other)
        self.assertEqual(result, False)
if __name__ == '__main__':
    unittest.main()