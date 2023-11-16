import os
import unittest

class TestCase(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        pass

    def verify_result(self):
        if False:
            return 10
        pass

    def test_cmd_run(self):
        if False:
            while True:
                i = 10
        pipeline = os.popen('pocsuite -k ecshop --options')
        res = pipeline.buffer.read().decode('utf-8')
        self.assertTrue('You can select dict_keys' in res)