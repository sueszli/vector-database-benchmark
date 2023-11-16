from __future__ import absolute_import
import unittest2
from st2client.utils.misc import merge_dicts

class MiscUtilTestCase(unittest2.TestCase):

    def test_merge_dicts(self):
        if False:
            print('Hello World!')
        d1 = {'a': 1}
        d2 = {'a': 2}
        expected = {'a': 2}
        result = merge_dicts(d1, d2)
        self.assertEqual(result, expected)
        d1 = {'a': 1}
        d2 = {'b': 1}
        expected = {'a': 1, 'b': 1}
        result = merge_dicts(d1, d2)
        self.assertEqual(result, expected)
        d1 = {'a': 1}
        d2 = {'a': 3, 'b': 1}
        expected = {'a': 3, 'b': 1}
        result = merge_dicts(d1, d2)
        self.assertEqual(result, expected)
        d1 = {'a': 1, 'm': None}
        d2 = {'a': None, 'b': 1, 'c': None}
        expected = {'a': 1, 'b': 1, 'c': None, 'm': None}
        result = merge_dicts(d1, d2)
        self.assertEqual(result, expected)
        d1 = {'a': 1, 'b': {'a': 1, 'b': 2, 'c': 3}}
        d2 = {'b': {'b': 100}}
        expected = {'a': 1, 'b': {'a': 1, 'b': 100, 'c': 3}}
        result = merge_dicts(d1, d2)
        self.assertEqual(result, expected)