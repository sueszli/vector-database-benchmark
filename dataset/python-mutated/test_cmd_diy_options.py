import os
import unittest

class TestCase(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        pass

    def tearDown(self):
        if False:
            return 10
        pass

    def verify_result(self):
        if False:
            print('Hello World!')
        pass

    def test_cmd_run(self):
        if False:
            return 10
        path = os.path.dirname(os.path.realpath(__file__))
        eval_path = os.path.join(path, '../pocsuite3/cli.py')
        poc_path = os.path.join(path, 'login_demo.py')
        command = f'python3 {eval_path} -u https://example.com -r {poc_path} --verify -v 2  --password mypass123 --username "asd asd" --testt abctest'
        pipeline = os.popen(command)
        res = pipeline.buffer.read().decode('utf-8')
        self.assertTrue('1 / 1' in res)