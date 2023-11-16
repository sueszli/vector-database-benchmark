import unittest
from cupy import _core

class TestArrayOwndata(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.a = _core.ndarray(())

    def test_original_array(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.a.flags.owndata is True

    def test_view_array(self):
        if False:
            while True:
                i = 10
        v = self.a.view()
        assert v.flags.owndata is False

    def test_reshaped_array(self):
        if False:
            print('Hello World!')
        r = self.a.reshape(())
        assert r.flags.owndata is False