import unittest
from tests.support.my_mock import MyMock

class TestMyMock(unittest.TestCase):

    def test_existing_method(self):
        if False:
            for i in range(10):
                print('nop')
        a = MyMock({'bar': lambda : 'bar-value'})
        assert 'bar-value' == a.bar()

    def test_not_existing_method(self):
        if False:
            i = 10
            return i + 15
        a = MyMock()
        self.assertRaises(NotImplementedError, lambda : a.non_existing())