import unittest
import numpy as np
from test_softmax_op import ref_softmax
import paddle
from paddle.base import core

class TestSoftmax2DAPI(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.shape = [2, 6, 5, 4]
        self.x_np = np.random.uniform(-1, 1, self.shape).astype('float64')
        self.axis = -3
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() else paddle.CPUPlace()

    def test_static_api(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
            m = paddle.nn.Softmax2D()
            out = m(x)
            exe = paddle.static.Executor(self.place)
            (res,) = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = ref_softmax(self.x_np, self.axis)
        np.testing.assert_allclose(out_ref, res, rtol=1e-05)

    def test_dygraph_api(self):
        if False:
            return 10
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.Softmax2D()
        out = m(x)
        out_ref = ref_softmax(self.x_np, self.axis)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)
        paddle.enable_static()

class TestSoftmax2DShape(TestSoftmax2DAPI):

    def setUp(self):
        if False:
            print('Hello World!')
        self.shape = [2, 6, 4]
        self.x_np = np.random.uniform(-1, 1, self.shape).astype('float64')
        self.axis = -3
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() else paddle.CPUPlace()

class TestSoftmax2DFloat32(TestSoftmax2DAPI):

    def setUp(self):
        if False:
            return 10
        self.shape = [2, 3, 4]
        self.x_np = np.random.uniform(-1, 1, self.shape).astype('float32')
        self.axis = -3
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() else paddle.CPUPlace()

class TestSoftmax2DCPU(TestSoftmax2DAPI):

    def setUp(self):
        if False:
            return 10
        self.shape = [2, 6, 4]
        self.x_np = np.random.uniform(-1, 1, self.shape).astype('float64')
        self.axis = -3
        self.place = paddle.CPUPlace()

class TestSoftmax2DRepr(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() else paddle.CPUPlace()

    def test_extra_repr(self):
        if False:
            while True:
                i = 10
        paddle.disable_static(self.place)
        m = paddle.nn.Softmax2D(name='test')
        self.assertTrue(m.extra_repr() == 'name=test')
        paddle.enable_static()

class TestSoftmax2DError(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() else paddle.CPUPlace()

    def test_static_error(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', [5, 5], 'float32')
            m = paddle.nn.Softmax2D()
            self.assertRaises(AssertionError, m, x)

    def test_dygraph_error(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static(self.place)
        x_np = np.random.randn(2, 3, 4, 2, 3)
        x = paddle.to_tensor(x_np, dtype='float64')
        m = paddle.nn.Softmax2D()
        self.assertRaises(AssertionError, m, x)
if __name__ == '__main__':
    unittest.main()