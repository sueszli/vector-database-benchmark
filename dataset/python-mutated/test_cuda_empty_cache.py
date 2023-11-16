import unittest
import paddle

class TestEmptyCache(unittest.TestCase):

    def test_empty_cache(self):
        if False:
            i = 10
            return i + 15
        x = paddle.randn((2, 10, 12)).astype('float32')
        del x
        self.assertIsNone(paddle.device.cuda.empty_cache())
if __name__ == '__main__':
    unittest.main()