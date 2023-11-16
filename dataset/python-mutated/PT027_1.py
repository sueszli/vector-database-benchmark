import unittest
import pytest

class Test(unittest.TestCase):

    def test_pytest_raises(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValueError):
            raise ValueError

    def test_errors(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            raise ValueError