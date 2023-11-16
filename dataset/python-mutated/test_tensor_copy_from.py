import unittest
import numpy as np
import paddle
from paddle.base.core import LoDTensor as Tensor

class TestTensorCopyFrom(unittest.TestCase):

    def test_main(self):
        if False:
            return 10
        place = paddle.CPUPlace()
        np_value = np.random.random(size=[10, 30]).astype('float32')
        t_src = Tensor()
        t_src.set(np_value, place)
        np.testing.assert_array_equal(np_value, t_src)
        t_dst1 = Tensor()
        t_dst1._copy_from(t_src, place)
        np.testing.assert_array_equal(np_value, t_dst1)
        t_dst2 = Tensor()
        t_dst2._copy_from(t_src, place, 5)
        np.testing.assert_array_equal(np.array(np_value[0:5]), t_dst2)
if __name__ == '__main__':
    unittest.main()