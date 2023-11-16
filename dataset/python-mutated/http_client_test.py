"""Unit tests for the http_client module."""
import os
import unittest
import mock
from httplib2 import ProxyInfo
from apache_beam.internal.http_client import DEFAULT_HTTP_TIMEOUT_SECONDS
from apache_beam.internal.http_client import get_new_http
from apache_beam.internal.http_client import proxy_info_from_environment_var

class HttpClientTest(unittest.TestCase):

    def test_proxy_from_env_http_with_port(self):
        if False:
            while True:
                i = 10
        with mock.patch.dict(os.environ, http_proxy='http://localhost:9000'):
            proxy_info = proxy_info_from_environment_var('http_proxy')
            expected = ProxyInfo(3, 'localhost', 9000)
            self.assertEqual(str(expected), str(proxy_info))

    def test_proxy_from_env_https_with_port(self):
        if False:
            for i in range(10):
                print('nop')
        with mock.patch.dict(os.environ, https_proxy='https://localhost:9000'):
            proxy_info = proxy_info_from_environment_var('https_proxy')
            expected = ProxyInfo(3, 'localhost', 9000)
            self.assertEqual(str(expected), str(proxy_info))

    def test_proxy_from_env_http_without_port(self):
        if False:
            return 10
        with mock.patch.dict(os.environ, http_proxy='http://localhost'):
            proxy_info = proxy_info_from_environment_var('http_proxy')
            expected = ProxyInfo(3, 'localhost', 80)
            self.assertEqual(str(expected), str(proxy_info))

    def test_proxy_from_env_https_without_port(self):
        if False:
            return 10
        with mock.patch.dict(os.environ, https_proxy='https://localhost'):
            proxy_info = proxy_info_from_environment_var('https_proxy')
            expected = ProxyInfo(3, 'localhost', 443)
            self.assertEqual(str(expected), str(proxy_info))

    def test_proxy_from_env_http_without_method(self):
        if False:
            print('Hello World!')
        with mock.patch.dict(os.environ, http_proxy='localhost:8000'):
            proxy_info = proxy_info_from_environment_var('http_proxy')
            expected = ProxyInfo(3, 'localhost', 8000)
            self.assertEqual(str(expected), str(proxy_info))

    def test_proxy_from_env_https_without_method(self):
        if False:
            while True:
                i = 10
        with mock.patch.dict(os.environ, https_proxy='localhost:8000'):
            proxy_info = proxy_info_from_environment_var('https_proxy')
            expected = ProxyInfo(3, 'localhost', 8000)
            self.assertEqual(str(expected), str(proxy_info))

    def test_proxy_from_env_http_without_port_without_method(self):
        if False:
            for i in range(10):
                print('nop')
        with mock.patch.dict(os.environ, http_proxy='localhost'):
            proxy_info = proxy_info_from_environment_var('http_proxy')
            expected = ProxyInfo(3, 'localhost', 80)
            self.assertEqual(str(expected), str(proxy_info))

    def test_proxy_from_env_https_without_port_without_method(self):
        if False:
            return 10
        with mock.patch.dict(os.environ, https_proxy='localhost'):
            proxy_info = proxy_info_from_environment_var('https_proxy')
            expected = ProxyInfo(3, 'localhost', 443)
            self.assertEqual(str(expected), str(proxy_info))

    def test_proxy_from_env_invalid_var(self):
        if False:
            i = 10
            return i + 15
        proxy_info = proxy_info_from_environment_var('http_proxy_host')
        expected = None
        self.assertEqual(str(expected), str(proxy_info))

    def test_proxy_from_env_wrong_method_in_var_name(self):
        if False:
            while True:
                i = 10
        with mock.patch.dict(os.environ, smtp_proxy='localhost'):
            with self.assertRaises(KeyError):
                proxy_info_from_environment_var('smtp_proxy')

    def test_proxy_from_env_wrong_method_in_url(self):
        if False:
            print('Hello World!')
        with mock.patch.dict(os.environ, http_proxy='smtp://localhost:8000'):
            proxy_info = proxy_info_from_environment_var('http_proxy')
            expected = ProxyInfo(3, 'smtp', 80)
            self.assertEqual(str(expected), str(proxy_info))

    def test_get_new_http_proxy_info(self):
        if False:
            i = 10
            return i + 15
        with mock.patch.dict(os.environ, http_proxy='localhost'):
            http = get_new_http()
            expected = ProxyInfo(3, 'localhost', 80)
            self.assertEqual(str(http.proxy_info), str(expected))

    def test_get_new_http_timeout(self):
        if False:
            return 10
        http = get_new_http()
        self.assertEqual(http.timeout, DEFAULT_HTTP_TIMEOUT_SECONDS)
if __name__ == '__main__':
    unittest.main()