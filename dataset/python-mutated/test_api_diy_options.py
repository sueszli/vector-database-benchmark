import os
import unittest
from pocsuite3.api import init_pocsuite
from pocsuite3.api import start_pocsuite
from pocsuite3.api import get_results
from pocsuite3.api import paths

class TestCase(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        pass

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def verify_result(self):
        if False:
            i = 10
            return i + 15
        config = {'url': ['https://www.baidu.com/'], 'poc': [os.path.join(paths.POCSUITE_ROOT_PATH, '../tests/login_demo.py')], 'username': 'asd', 'password': 'asdss', 'verbose': 0, 'timeout': 10}
        init_pocsuite(config)
        start_pocsuite()
        result = get_results().pop()
        self.assertTrue(result.status == 'success')

    @unittest.skip(reason='significant latency')
    def test_cookie(self):
        if False:
            return 10
        config = {'url': ['http://httpbin.org/post'], 'poc': [os.path.join(paths.POCSUITE_ROOT_PATH, '../tests/login_demo.py')], 'username': 'asd', 'password': 'asdss', 'cookie': 'test=1', 'verbose': 0, 'timeout': 10}
        init_pocsuite(config)
        start_pocsuite()
        result = get_results().pop()
        self.assertTrue(result.status == 'success')

    @unittest.skip(reason='significant latency')
    def test_cookie_dict_params(self):
        if False:
            for i in range(10):
                print('nop')
        config = {'url': ['http://httpbin.org/post'], 'poc': [os.path.join(paths.POCSUITE_ROOT_PATH, '../tests/login_demo.py')], 'username': 'asd', 'password': 'asdss', 'cookie': {'test': '123'}, 'verbose': 0, 'timeout': 10}
        init_pocsuite(config)
        start_pocsuite()
        result = get_results().pop()
        self.assertTrue(result.status == 'success')

    def test_import_run(self):
        if False:
            while True:
                i = 10
        self.verify_result()