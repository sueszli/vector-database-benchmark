import unittest
from pocsuite3.api import init_pocsuite
from pocsuite3.api import start_pocsuite
from pocsuite3.api import get_results

class TestCase(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.config = {'url': 'http://127.0.0.1:8080', 'poc': 'ecshop_rce'}
        init_pocsuite(self.config)

    def tearDown(self):
        if False:
            print('Hello World!')
        pass

    def verify_result(self):
        if False:
            return 10
        result = get_results().pop()
        self.assertTrue(result[5] == 'success')

    @unittest.skip(reason='test')
    def test_import_run(self):
        if False:
            return 10
        start_pocsuite()
        self.verify_result()