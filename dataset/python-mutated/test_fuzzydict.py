"""Tests for class FuzzyDict"""
import unittest
import sys
from collections import OrderedDict
sys.path.append('.')
from pywinauto.fuzzydict import FuzzyDict

class FuzzyTestCase(unittest.TestCase):
    """Perform some tests"""
    test_dict = OrderedDict([(u'Hiya', 1), (u'hiyä', 2), (u'test3', 3), (1, 324)])

    def test_creation_empty(self):
        if False:
            for i in range(10):
                print('nop')
        'Verify that not specifying any values creates an empty dictionary'
        fd = FuzzyDict()
        self.assertEqual(fd, {})

    def test_creation_dict(self):
        if False:
            i = 10
            return i + 15
        'Test creating a fuzzy dict'
        fd = FuzzyDict(self.test_dict)
        self.assertEqual(fd, self.test_dict)
        self.assertEqual(self.test_dict[u'Hiya'], fd[u'hiya'])
        fd2 = FuzzyDict(self.test_dict, cutoff=0.8)
        self.assertEqual(fd, self.test_dict)
        self.assertRaises(KeyError, fd2.__getitem__, u'hiya')

    def test_contains(self):
        if False:
            return 10
        'Test checking if an item is in a FuzzyDict'
        fd = FuzzyDict(self.test_dict)
        self.assertEqual(True, fd.__contains__(u'hiya'))
        self.assertEqual(True, fd.__contains__(u'test3'))
        self.assertEqual(True, fd.__contains__(u'hiyä'))
        self.assertEqual(False, fd.__contains__(u'FuzzyWuzzy'))
        self.assertEqual(True, fd.__contains__(1))
        self.assertEqual(False, fd.__contains__(23))

    def test_get_item(self):
        if False:
            while True:
                i = 10
        'Test getting items from a FuzzyDict'
        fd = FuzzyDict(self.test_dict)
        self.assertEqual(self.test_dict[u'Hiya'], fd[u'hiya'])
        self.assertRaises(KeyError, fd.__getitem__, u'FuzzyWuzzy')
        fd2 = FuzzyDict(self.test_dict, cutoff=0.14)
        self.assertEqual(1, fd2[u'FuzzyWuzzy'])
        self.assertEqual(324, fd2[1])
        self.assertRaises(KeyError, fd2.__getitem__, 23)
if __name__ == '__main__':
    unittest.main()