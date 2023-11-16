"""setUpModule is always called, even if all tests in the module are skipped"""
import unittest

def setUpModule():
    if False:
        return 10
    assert 0

@unittest.skip('skip all tests')
class Base(unittest.TestCase):

    def test(self):
        if False:
            i = 10
            return i + 15
        assert 0