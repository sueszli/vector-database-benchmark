import unittest
import win32com.client
import win32com.server.util
import win32com.test.util

class Tester:
    _public_methods_ = ['TestValue']

    def TestValue(self, v):
        if False:
            return 10
        pass

def test_ob():
    if False:
        i = 10
        return i + 15
    return win32com.client.Dispatch(win32com.server.util.wrap(Tester()))

class TestException(Exception):
    pass

class BadConversions:

    def __float__(self):
        if False:
            while True:
                i = 10
        raise TestException

class TestCase(win32com.test.util.TestCase):

    def test_float(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            test_ob().TestValue(BadConversions())
            raise Exception('Should not have worked')
        except Exception as e:
            assert isinstance(e, TestException)
if __name__ == '__main__':
    unittest.main()