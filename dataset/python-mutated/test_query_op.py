import unittest
import paddle
from paddle.base import core

class TestCudnnVersion(unittest.TestCase):

    def test_no_cudnn(self):
        if False:
            for i in range(10):
                print('nop')
        cudnn_version = paddle.get_cudnn_version()
        if not core.is_compiled_with_cuda():
            self.assertEqual(cudnn_version is None, True)
        else:
            self.assertEqual(isinstance(cudnn_version, int), True)
if __name__ == '__main__':
    unittest.main()