from __future__ import absolute_import
import unittest
from st2common.util import mongoescape

class TestMongoEscape(unittest.TestCase):

    def test_unnested(self):
        if False:
            while True:
                i = 10
        field = {'k1.k1.k1': 'v1', 'k2$': 'v2', '$k3.': 'v3'}
        escaped = mongoescape.escape_chars(field)
        self.assertEqual(escaped, {'k1．k1．k1': 'v1', 'k2＄': 'v2', '＄k3．': 'v3'}, 'Escaping failed.')
        unescaped = mongoescape.unescape_chars(escaped)
        self.assertEqual(unescaped, field, 'Unescaping failed.')

    def test_nested(self):
        if False:
            for i in range(10):
                print('nop')
        nested_field = {'nk1.nk1.nk1': 'v1', 'nk2$': 'v2', '$nk3.': 'v3'}
        field = {'k1.k1.k1': nested_field, 'k2$': 'v2', '$k3.': 'v3'}
        escaped = mongoescape.escape_chars(field)
        self.assertEqual(escaped, {'k1．k1．k1': {'＄nk3．': 'v3', 'nk1．nk1．nk1': 'v1', 'nk2＄': 'v2'}, 'k2＄': 'v2', '＄k3．': 'v3'}, 'un-escaping failed.')
        unescaped = mongoescape.unescape_chars(escaped)
        self.assertEqual(unescaped, field, 'Unescaping failed.')

    def test_unescaping_of_rule_criteria(self):
        if False:
            print('Hello World!')
        escaped = {'k1․k1․k1': 'v1', 'k2$': 'v2', '$k3․': 'v3'}
        unescaped = {'k1.k1.k1': 'v1', 'k2$': 'v2', '$k3.': 'v3'}
        result = mongoescape.unescape_chars(escaped)
        self.assertEqual(result, unescaped)

    def test_original_value(self):
        if False:
            i = 10
            return i + 15
        field = {'k1.k2.k3': 'v1'}
        escaped = mongoescape.escape_chars(field)
        self.assertIn('k1.k2.k3', list(field.keys()))
        self.assertIn('k1．k2．k3', list(escaped.keys()))
        unescaped = mongoescape.unescape_chars(escaped)
        self.assertIn('k1.k2.k3', list(unescaped.keys()))
        self.assertIn('k1．k2．k3', list(escaped.keys()))

    def test_complex(self):
        if False:
            for i in range(10):
                print('nop')
        field = {'k1.k2': [{'l1.l2': '123'}, {'l3.l4': '456'}], 'k3': [{'l5.l6': '789'}], 'k4.k5': [1, 2, 3], 'k6': ['a', 'b']}
        expected = {'k1．k2': [{'l1．l2': '123'}, {'l3．l4': '456'}], 'k3': [{'l5．l6': '789'}], 'k4．k5': [1, 2, 3], 'k6': ['a', 'b']}
        escaped = mongoescape.escape_chars(field)
        self.assertDictEqual(expected, escaped)
        unescaped = mongoescape.unescape_chars(escaped)
        self.assertDictEqual(field, unescaped)

    def test_complex_list(self):
        if False:
            while True:
                i = 10
        field = [{'k1.k2': [{'l1.l2': '123'}, {'l3.l4': '456'}]}, {'k3': [{'l5.l6': '789'}]}, {'k4.k5': [1, 2, 3]}, {'k6': ['a', 'b']}]
        expected = [{'k1．k2': [{'l1．l2': '123'}, {'l3．l4': '456'}]}, {'k3': [{'l5．l6': '789'}]}, {'k4．k5': [1, 2, 3]}, {'k6': ['a', 'b']}]
        escaped = mongoescape.escape_chars(field)
        self.assertListEqual(expected, escaped)
        unescaped = mongoescape.unescape_chars(escaped)
        self.assertListEqual(field, unescaped)