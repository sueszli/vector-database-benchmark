import unittest
import paddle
from paddle.device import xpu

class TestSynchronize(unittest.TestCase):

    def test_synchronize(self):
        if False:
            while True:
                i = 10
        if paddle.is_compiled_with_xpu():
            self.assertIsNone(xpu.synchronize())
            self.assertIsNone(xpu.synchronize(0))
            self.assertIsNone(xpu.synchronize(paddle.XPUPlace(0)))
            self.assertRaises(ValueError, xpu.synchronize, 'xpu:0')
if __name__ == '__main__':
    unittest.main()