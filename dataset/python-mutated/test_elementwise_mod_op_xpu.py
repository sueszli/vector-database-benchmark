import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test import OpTest
from op_test_xpu import XPUOpTest
import paddle
from paddle import base
paddle.enable_static()

class XPUTestElementwiseModOp(XPUOpTestWrapper):

    def __init__(self) -> None:
        if False:
            return 10
        self.op_name = 'elementwise_mod'
        self.use_dynamic_create_class = False

    class ElementwiseModOp(XPUOpTest):

        def init_kernel_type(self):
            if False:
                return 10
            self.use_mkldnn = False

        def init_input_output(self):
            if False:
                while True:
                    i = 10
            self.x = np.random.uniform(0, 10000, [10, 10]).astype(self.dtype)
            self.y = np.random.uniform(0, 1000, [10, 10]).astype(self.dtype)
            self.out = np.mod(self.x, self.y)
            self.inputs = {'X': OpTest.np_dtype_to_base_dtype(self.x), 'Y': OpTest.np_dtype_to_base_dtype(self.y)}
            self.outputs = {'Out': self.out}
            self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}

        def init_dtype(self):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def init_axis(self):
            if False:
                while True:
                    i = 10
            pass

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            self.op_type = 'elementwise_mod'
            self.use_xpu = True
            self.dtype = self.in_type
            self.axis = -1
            self.init_dtype()
            self.init_input_output()
            self.init_kernel_type()
            self.init_axis()

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

    class TestRemainderOp(unittest.TestCase):

        def test_dygraph(self):
            if False:
                while True:
                    i = 10
            with base.dygraph.guard():
                np_x = np.random.rand(22, 128, 3).astype('int64')
                np_y = np.random.rand(22, 128, 3).astype('int64')
                x = paddle.to_tensor(np_x)
                y = paddle.to_tensor(np_y)
                z = paddle.remainder(x, y)
                np_z = z.numpy()
                z_expected = np.mod(np_x, np_y)
                self.assertEqual((np_z == z_expected).all(), True)
                np_x = np.array([-3.3, 11.5, -2, 3.5])
                np_y = np.array([-1.2, 2.0, 3.3, -2.3])
                x = paddle.to_tensor(np_x)
                y = paddle.to_tensor(np_y)
                z = x % y
                z_expected = np.array([-0.9, 1.5, 1.3, -1.1])
                np.testing.assert_allclose(z_expected, z.numpy(), rtol=1e-05)
                np_x = np.random.rand(22, 128, 3).astype('int32')
                np_y = np.random.rand(22, 128, 3).astype('int32')
                x = paddle.to_tensor(np_x)
                y = paddle.to_tensor(np_y)
                z = paddle.remainder(x, y)
                np_z = z.numpy()
                z_expected = np.mod(np_x, np_y)
                self.assertEqual((np_z == z_expected).all(), True)
                np_x = np.array([-3, 11, -2, 3])
                np_y = np.array([-1, 2, 3, -2])
                x = paddle.to_tensor(np_x, dtype='float16')
                y = paddle.to_tensor(np_y, dtype='float16')
                z = x % y
                z_expected = np.array([0, 1, 1, -1])
                np.testing.assert_allclose(z_expected, z.numpy(), rtol=1e-05)
support_types = get_xpu_op_support_types('elementwise_mod')
for stype in support_types:
    create_test_class(globals(), XPUTestElementwiseModOp, stype)
if __name__ == '__main__':
    unittest.main()