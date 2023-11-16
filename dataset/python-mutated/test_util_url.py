from __future__ import absolute_import
import unittest2
from st2common.util.url import get_url_without_trailing_slash
from six.moves import zip

class URLUtilsTestCase(unittest2.TestCase):

    def test_get_url_without_trailing_slash(self):
        if False:
            print('Hello World!')
        values = ['http://localhost:1818/foo/bar/', 'http://localhost:1818/foo/bar', 'http://localhost:1818/', 'http://localhost:1818']
        expected = ['http://localhost:1818/foo/bar', 'http://localhost:1818/foo/bar', 'http://localhost:1818', 'http://localhost:1818']
        for (value, expected_result) in zip(values, expected):
            actual = get_url_without_trailing_slash(value=value)
            self.assertEqual(actual, expected_result)