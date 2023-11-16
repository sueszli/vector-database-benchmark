from __future__ import absolute_import
import unittest2
from st2common.util.http import parse_content_type_header
from six.moves import zip
__all__ = ['HTTPUtilTestCase']

class HTTPUtilTestCase(unittest2.TestCase):

    def test_parse_content_type_header(self):
        if False:
            for i in range(10):
                print('nop')
        values = ['application/json', 'foo/bar', 'application/json; charset=utf-8', 'application/json; charset=utf-8; foo=bar']
        expected_results = [('application/json', {}), ('foo/bar', {}), ('application/json', {'charset': 'utf-8'}), ('application/json', {'charset': 'utf-8', 'foo': 'bar'})]
        for (value, expected_result) in zip(values, expected_results):
            result = parse_content_type_header(content_type=value)
            self.assertEqual(result, expected_result)