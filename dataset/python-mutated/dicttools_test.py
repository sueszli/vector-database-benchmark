import unittest
from typing import Any, Dict
from parameterized import parameterized
from streamlit.elements.lib.dicttools import remove_none_values

class DictToolsTest(unittest.TestCase):

    @parameterized.expand([({}, {}), ({'a': 1, 'b': 2}, {'a': 1, 'b': 2}), ({'a': 1, 'b': None}, {'a': 1}), ({'a': 1, 'b': {'c': None}}, {'a': 1, 'b': {}}), ({'a': 1, 'b': {'c': 2}}, {'a': 1, 'b': {'c': 2}}), ({'a': 1, 'b': {'c': None, 'd': 3}}, {'a': 1, 'b': {'d': 3}})])
    def test_remove_none_values(self, input: Dict[str, Any], expected: Dict[str, Any]):
        if False:
            while True:
                i = 10
        'Test remove_none_values.'
        self.assertEqual(remove_none_values(input), expected, f'Expected {input} to be transformed into {expected}.')