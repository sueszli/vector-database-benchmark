from __future__ import absolute_import
import json
import unittest2
from st2common.util.casts import get_cast

class CastsTestCase(unittest2.TestCase):

    def test_cast_string(self):
        if False:
            print('Hello World!')
        cast_func = get_cast('string')
        value = 'test1'
        result = cast_func(value)
        self.assertEqual(result, 'test1')
        value = 'test2'
        result = cast_func(value)
        self.assertEqual(result, 'test2')
        value = ''
        result = cast_func(value)
        self.assertEqual(result, '')
        value = None
        result = cast_func(value)
        self.assertEqual(result, None)
        value = []
        expected_msg = 'Value "\\[\\]" must either be a string or None. Got "list"'
        self.assertRaisesRegexp(ValueError, expected_msg, cast_func, value)

    def test_cast_array(self):
        if False:
            return 10
        cast_func = get_cast('array')
        value = str([1, 2, 3])
        result = cast_func(value)
        self.assertEqual(result, [1, 2, 3])
        value = json.dumps([4, 5, 6])
        result = cast_func(value)
        self.assertEqual(result, [4, 5, 6])
        value = '\\invalid'
        self.assertRaises(SyntaxError, cast_func, value)