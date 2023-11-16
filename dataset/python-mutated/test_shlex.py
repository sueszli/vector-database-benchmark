from __future__ import annotations
import unittest
from ansible.utils.shlex import shlex_split

class TestSplit(unittest.TestCase):

    def test_trivial(self):
        if False:
            return 10
        self.assertEqual(shlex_split('a b c'), ['a', 'b', 'c'])

    def test_unicode(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(shlex_split(u'a b č'), [u'a', u'b', u'č'])

    def test_quoted(self):
        if False:
            print('Hello World!')
        self.assertEqual(shlex_split('"a b" c'), ['a b', 'c'])

    def test_comments(self):
        if False:
            while True:
                i = 10
        self.assertEqual(shlex_split('"a b" c # d', comments=True), ['a b', 'c'])

    def test_error(self):
        if False:
            while True:
                i = 10
        self.assertRaises(ValueError, shlex_split, 'a "b')