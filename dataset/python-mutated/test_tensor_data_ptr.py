import unittest
import numpy as np
import paddle

class TestTensorDataPtr(unittest.TestCase):

    def test_tensor_data_ptr(self):
        if False:
            while True:
                i = 10
        np_src = np.random.random((3, 8, 8))
        src = paddle.to_tensor(np_src, dtype='float64')
        dst = paddle.Tensor()
        src._share_buffer_to(dst)
        self.assertTrue(src.data_ptr() is not None)
        self.assertEqual(src.data_ptr(), dst.data_ptr())
if __name__ == '__main__':
    unittest.main()