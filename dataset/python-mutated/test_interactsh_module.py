import unittest
from pocsuite3.api import Interactsh, requests

class TestCase(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def tearDown(self):
        if False:
            return 10
        pass

    @unittest.skip(reason='interactsh service is unstable')
    def test_interactsh(self):
        if False:
            for i in range(10):
                print('nop')
        ISH = Interactsh(token='', server='')
        (url, flag) = ISH.build_request(method='https')
        requests.get(url, timeout=5, verify=False)
        self.assertTrue(ISH.verify(flag))
if __name__ == '__main__':
    unittest.main()