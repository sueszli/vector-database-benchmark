import os
import unittest
from slack_sdk.proxy_env_variable_loader import load_http_proxy_from_env
from tests.helpers import remove_os_env_temporarily, restore_os_env

class TestProxyEnvVariableLoader(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.old_env = remove_os_env_temporarily()

    def tearDown(self):
        if False:
            while True:
                i = 10
        os.environ.clear()
        restore_os_env(self.old_env)

    def test_load_lower_case(self):
        if False:
            i = 10
            return i + 15
        os.environ['https_proxy'] = 'http://localhost:9999'
        url = load_http_proxy_from_env()
        self.assertEqual(url, 'http://localhost:9999')

    def test_load_upper_case(self):
        if False:
            return 10
        os.environ['HTTPS_PROXY'] = 'http://localhost:9999'
        url = load_http_proxy_from_env()
        self.assertEqual(url, 'http://localhost:9999')

    def test_load_all_empty_case(self):
        if False:
            for i in range(10):
                print('nop')
        os.environ['HTTP_PROXY'] = ''
        os.environ['http_proxy'] = ''
        os.environ['HTTPS_PROXY'] = ''
        os.environ['https_proxy'] = ''
        url = load_http_proxy_from_env()
        self.assertEqual(url, None)

    def test_proxy_url_is_none_case(self):
        if False:
            for i in range(10):
                print('nop')
        os.environ.pop('HTTPS_PROXY', None)
        os.environ.pop('https_proxy', None)
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('http_proxy', None)
        url = load_http_proxy_from_env()
        self.assertEqual(url, None)