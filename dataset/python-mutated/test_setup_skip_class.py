"""Skipping an entire subclass with unittest.skip() should *not* call setUpClass from a base class."""
import unittest

class Base(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        assert 0

@unittest.skip('skip all tests')
class Test(Base):

    def test_foo(self):
        if False:
            for i in range(10):
                print('nop')
        assert 0