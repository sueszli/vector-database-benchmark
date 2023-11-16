import unittest
import numpy as np
import paddle
from paddle.base import core

def fn(x, shape):
    if False:
        i = 10
        return i + 15
    out = paddle.expand(x, shape=shape)
    return out

class TestIntarrayInput(unittest.TestCase):
    """This case is set to test int_array input process during composite rule."""

    def test_non_tensor_input(self):
        if False:
            return 10
        core._set_prim_all_enabled(True)
        np_data = np.random.random([3, 4]).astype('float32')
        tensor_data = paddle.to_tensor(np_data)
        net = paddle.jit.to_static(fn)
        _ = net(tensor_data, shape=[2, 3, 4]).numpy()
        core._set_prim_all_enabled(False)

    def test_error_input(self):
        if False:
            return 10
        'In composite rules, tensor shape is not supported in int_array input'
        core._set_prim_all_enabled(True)
        np_data = np.random.random([3, 4]).astype('float32')
        tensor_data = paddle.to_tensor(np_data)
        shape = paddle.to_tensor([2, 3, 4])
        net = paddle.jit.to_static(fn, full_graph=True)
        with self.assertRaises(NotImplementedError):
            _ = net(tensor_data, shape).numpy()
        core._set_prim_all_enabled(False)
if __name__ == '__main__':
    unittest.main()