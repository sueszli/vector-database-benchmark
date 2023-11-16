import unittest
from cupy import cuda

@unittest.skipUnless(cuda.nvtx.available, 'nvtx is not installed')
class TestNVTX(unittest.TestCase):

    def test_Mark(self):
        if False:
            for i in range(10):
                print('nop')
        cuda.nvtx.Mark('test:Mark', 0)

    def test_MarkC(self):
        if False:
            i = 10
            return i + 15
        cuda.nvtx.MarkC('test:MarkC', 4278190080)

    def test_RangePush(self):
        if False:
            print('Hello World!')
        cuda.nvtx.RangePush('test:RangePush', 1)
        cuda.nvtx.RangePop()

    def test_RangePushC(self):
        if False:
            i = 10
            return i + 15
        cuda.nvtx.RangePushC('test:RangePushC', 4278190080)
        cuda.nvtx.RangePop()