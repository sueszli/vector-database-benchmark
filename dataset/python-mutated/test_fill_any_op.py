import unittest
import numpy as np
from op_test import OpTest
import paddle

def fill_any_wrapper(x, value=0):
    if False:
        for i in range(10):
            print('nop')
    return paddle._legacy_C_ops.fill_any(x, 'value', value)

class TestFillAnyOp(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'fill_any'
        self.python_api = fill_any_wrapper
        self.dtype = 'float64'
        self.value = 0.0
        self.init()
        self.inputs = {'X': np.random.random((20, 30)).astype(self.dtype)}
        self.attrs = {'value': float(self.value)}
        self.outputs = {'Out': self.value * np.ones_like(self.inputs['X']).astype(self.dtype)}

    def init(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_check_output(self):
        if False:
            return 10
        self.check_output()

    def test_check_grad(self):
        if False:
            return 10
        self.check_grad(['X'], 'Out')

class TestFillAnyOpFloat32(TestFillAnyOp):

    def init(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.float32
        self.value = 0.0

class TestFillAnyOpFloat16(TestFillAnyOp):

    def init(self):
        if False:
            print('Hello World!')
        self.dtype = np.float16

class TestFillAnyOpvalue1(TestFillAnyOp):

    def init(self):
        if False:
            return 10
        self.dtype = np.float32
        self.value = 111111555

class TestFillAnyOpvalue2(TestFillAnyOp):

    def init(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.float32
        self.value = 11111.1111

class TestFillAnyInplace(unittest.TestCase):

    def test_fill_any_version(self):
        if False:
            for i in range(10):
                print('nop')
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
            while True:
                i = 10
        with paddle.base.dygraph.guard():
            x = paddle.full([10, 10], -1.0, dtype='float32')
            x.stop_gradient = False
            y = 2 * x
            y.fill_(1)
            y.backward()
            np.testing.assert_array_equal(x.grad.numpy(), np.zeros([10, 10]))
if __name__ == '__main__':
    unittest.main()