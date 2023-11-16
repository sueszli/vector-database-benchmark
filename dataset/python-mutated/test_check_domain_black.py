import unittest
from app.utils import check_domain_black, is_valid_domain

class TestCDNName(unittest.TestCase):

    def test_black(self):
        if False:
            print('Hello World!')
        result = check_domain_black('test.wire.comm.example.com')
        self.assertTrue(result)
        result = check_domain_black('test.wire1.comm.example.com')
        self.assertFalse(result)

    def test_is_valid_domain(self):
        if False:
            i = 10
            return i + 15
        result = is_valid_domain('!test.test.example.com')
        self.assertFalse(result)
if __name__ == '__main__':
    unittest.main()