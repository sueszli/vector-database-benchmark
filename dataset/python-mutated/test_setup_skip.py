"""Skipping an entire subclass with unittest.skip() should *not* call setUp from a base class."""
import unittest

class Base(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        assert 0

@unittest.skip('skip all tests')
class Test(Base):

    def test_foo(self):
        if False:
            return 10
        assert 0