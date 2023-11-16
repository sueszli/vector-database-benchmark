from __future__ import absolute_import
import unittest2
from oslo_config import cfg
from st2common.constants.api import DEFAULT_API_VERSION
from st2common.util.api import get_base_public_api_url
from st2common.util.api import get_full_public_api_url
from st2tests.config import parse_args
from six.moves import zip
parse_args()

class APIUtilsTestCase(unittest2.TestCase):

    def test_get_base_public_api_url(self):
        if False:
            return 10
        values = ['http://foo.bar.com', 'http://foo.bar.com/', 'http://foo.bar.com:8080', 'http://foo.bar.com:8080/', 'http://localhost:8080/']
        expected = ['http://foo.bar.com', 'http://foo.bar.com', 'http://foo.bar.com:8080', 'http://foo.bar.com:8080', 'http://localhost:8080']
        for (mock_value, expected_result) in zip(values, expected):
            cfg.CONF.auth.api_url = mock_value
            actual = get_base_public_api_url()
            self.assertEqual(actual, expected_result)

    def test_get_full_public_api_url(self):
        if False:
            while True:
                i = 10
        values = ['http://foo.bar.com', 'http://foo.bar.com/', 'http://foo.bar.com:8080', 'http://foo.bar.com:8080/', 'http://localhost:8080/']
        expected = ['http://foo.bar.com/' + DEFAULT_API_VERSION, 'http://foo.bar.com/' + DEFAULT_API_VERSION, 'http://foo.bar.com:8080/' + DEFAULT_API_VERSION, 'http://foo.bar.com:8080/' + DEFAULT_API_VERSION, 'http://localhost:8080/' + DEFAULT_API_VERSION]
        for (mock_value, expected_result) in zip(values, expected):
            cfg.CONF.auth.api_url = mock_value
            actual = get_full_public_api_url()
            self.assertEqual(actual, expected_result)