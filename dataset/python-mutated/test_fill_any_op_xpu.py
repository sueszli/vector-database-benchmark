import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

class XPUTestFillAnyOp(XPUOpTestWrapper):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.op_name = 'fill_any'
        self.use_dynamic_create_class = False

    class TestFillAnyOp(XPUOpTest):

        def setUp(self):
            if False:
                print('Hello World!')
            self.op_type = 'fill_any'
            self.dtype = 'float64'
            self.value = 0.0
            self.init()
            self.inputs = {'X': np.random.random((20, 30)).astype(self.dtype)}
            self.attrs = {'value': float(self.value)}
            self.outputs = {'Out': self.value * np.ones_like(self.inputs['X']).astype(self.dtype)}

        def init(self):
            if False:
                return 10
            pass

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            self.check_output_with_place(paddle.XPUPlace(0))

        def test_check_grad(self):
            if False:
                print('Hello World!')
            self.check_grad_with_place(paddle.XPUPlace(0), ['X'], 'Out')

    class TestFillAnyOpFloat32(TestFillAnyOp):

        def init(self):
            if False:
                while True:
                    i = 10
            self.dtype = np.float32
            self.value = 0.0

    class TestFillAnyOpFloat16(TestFillAnyOp):

        def init(self):
            if False:
                i = 10
                return i + 15
            self.dtype = np.float16

    class TestFillAnyOpvalue1(TestFillAnyOp):

        def init(self):
            if False:
                i = 10
                return i + 15
            self.dtype = np.float32
            self.value = 111111555

    class TestFillAnyOpvalue2(TestFillAnyOp):

        def init(self):
            if False:
                return 10
            self.dtype = np.float32
            self.value = 11111.1111

    class TestFillAnyInplace(unittest.TestCase):

        def test_fill_any_version(self):
            if False:
                while True:
                    i = 10
            with paddle.base.dygraph.guard():
                var = paddle.to_tensor(np.ones((4, 2, 3)).astype(np.float32))
                self.assertEqual(var.inplace_version, 0)
                var.fill_(0)
                self.assertEqual(var.inplace_version, 1)
                var.fill_(0)
                self.assertEqual(var.inplace_version, 2)
                var.fill_(0)
                self.assertEqual(var.inplace_version, 3)

        def test_fill_any_eqaul(self):
            if False:
                while True:
                    i = 10
            with paddle.base.dygraph.guard():
                tensor = paddle.to_tensor(np.random.random((20, 30)).astype(np.float32))
                target = tensor.numpy()
                target[...] = 1
                tensor.fill_(1)
                self.assertEqual((tensor.numpy() == target).all().item(), True)

        def test_backward(self):
            if False:
                for i in range(10):
                    print('nop')
            with paddle.base.dygraph.guard():
                x = paddle.full([10, 10], -1.0, dtype='float32')
                x.stop_gradient = False
                y = 2 * x
                y.fill_(1)
                y.backward()
                np.testing.assert_array_equal(x.grad.numpy(), np.zeros([10, 10]))

class TestFillAnyLikeOpSpecialValue(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.special_values = [float('nan'), float('+inf'), float('-inf')]
        self.dtypes = ['float32', 'float16']

    def test_dygraph_api(self):
        if False:
            while True:
                i = 10
        paddle.disable_static()
        paddle.set_device('xpu')
        for dtype in self.dtypes:
            for value in self.special_values:
                ref = paddle.empty([4, 4], dtype=dtype)
                val_pd = paddle.full_like(ref, value, dtype=dtype)
                val_np = np.full([4, 4], value, dtype=dtype)
                np.testing.assert_equal(val_pd.numpy(), val_np)
        paddle.enable_static()
support_types = get_xpu_op_support_types('fill_any')
for stype in support_types:
    create_test_class(globals(), XPUTestFillAnyOp, stype)
if __name__ == '__main__':
    unittest.main()