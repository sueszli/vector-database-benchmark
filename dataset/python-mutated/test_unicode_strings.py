from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import six
import unittest
import turicreate as tc
import pytest
pytestmark = [pytest.mark.minimal]

class UnicodeStringTest(unittest.TestCase):

    def test_unicode_column_accessor(self):
        if False:
            i = 10
            return i + 15
        sf = tc.SFrame({'a': range(100)})
        self.assertEqual(sf[u'a'][0], sf['a'][0])

    def test_unicode_unpack_prefix(self):
        if False:
            for i in range(10):
                print('nop')
        sf = tc.SFrame({'a': [{'x': 1}, {'x': 2}, {'x': 3}]})
        sf = sf.unpack('a', u'ª')
        for col in sf.column_names():
            if six.PY2:
                self.assertTrue(col.startswith(u'ª'.encode('utf-8')))
            else:
                self.assertTrue(col.startswith(u'ª'))

    def test_unicode_column_construction(self):
        if False:
            return 10
        sf = tc.SFrame({u'ª': [1, 2, 3]})
        self.assertEqual(sf[u'ª'][0], 1)

    def test_access_nonexistent_column(self):
        if False:
            while True:
                i = 10
        sf = tc.SFrame({u'ª': [1, 2, 3], 'a': [4, 5, 6]})
        with self.assertRaises(RuntimeError):
            sf['b']
        with self.assertRaises(RuntimeError):
            sf[u'«']