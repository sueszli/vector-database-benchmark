import os
import sys
import unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test.helper import FakeYDL, is_download_test
from yt_dlp.extractor import IqiyiIE

class WarningLogger:

    def __init__(self):
        if False:
            print('Hello World!')
        self.messages = []

    def warning(self, msg):
        if False:
            print('Hello World!')
        self.messages.append(msg)

    def debug(self, msg):
        if False:
            for i in range(10):
                print('nop')
        pass

    def error(self, msg):
        if False:
            for i in range(10):
                print('nop')
        pass

@is_download_test
class TestIqiyiSDKInterpreter(unittest.TestCase):

    def test_iqiyi_sdk_interpreter(self):
        if False:
            while True:
                i = 10
        '\n        Test the functionality of IqiyiSDKInterpreter by trying to log in\n\n        If `sign` is incorrect, /validate call throws an HTTP 556 error\n        '
        logger = WarningLogger()
        ie = IqiyiIE(FakeYDL({'logger': logger}))
        ie._perform_login('foo', 'bar')
        self.assertTrue('unable to log in:' in logger.messages[0])
if __name__ == '__main__':
    unittest.main()