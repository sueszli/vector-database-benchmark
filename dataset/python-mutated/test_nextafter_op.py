import unittest
import numpy as np
from op_test import OpTest
import paddle

def ref_nextafter(x, y):
    if False:
        print('Hello World!')
    out = np.nextafter(x, y)
    return out

class TestNextafterAPI(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.x = np.random.rand(2, 3, 4, 5).astype('float32')
        self.y = np.random.rand(2, 3, 4, 5).astype('float32')
        self.x1 = np.array([0, 0, 10]).astype('float32')
        self.y1 = np.array([np.inf, -np.inf, 10]).astype('float32')
        self.x2 = np.random.rand(100).astype('float32')
        self.y2 = np.random.rand(100).astype('float32')
        self.place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()

    def test_static_api(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=self.x.shape, dtype='float32')
            y = paddle.static.data(name='y', shape=self.y.shape, dtype='float32')
            out = paddle.nextafter(x, y)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': self.x, 'y': self.y}, fetch_list=[out])
        out_ref = ref_nextafter(self.x, self.y)
        np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)
        with paddle.static.program_guard(paddle.static.Program()):
            x1 = paddle.static.data(name='x', shape=self.x1.shape, dtype='float32')
            y1 = paddle.static.data(name='y', shape=self.y1.shape, dtype='float32')
            out = paddle.nextafter(x1, y1)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': self.x1, 'y': self.y1}, fetch_list=[out])
        out_ref = ref_nextafter(self.x1, self.y1)
        np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)
        with paddle.static.program_guard(paddle.static.Program()):
            x2 = paddle.static.data(name='x', shape=self.x2.shape, dtype='float32')
            y2 = paddle.static.data(name='y', shape=self.y2.shape, dtype='float32')
            out = paddle.nextafter(x2, y2)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': self.x2, 'y': self.y2}, fetch_list=[out])
        out_ref = ref_nextafter(self.x2, self.y2)
        np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)

    def test_dygraph_api(self):
        if False:
            print('Hello World!')
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        out = paddle.nextafter(x, y)
        out_ref = ref_nextafter(self.x, self.y)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)
        paddle.enable_static()

class TestNextafterOP(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'nextafter'
        self.python_api = paddle.nextafter
        self.init_dtype()
        x = np.array([1, 2]).astype(self.dtype)
        y = np.array([2, 1]).astype(self.dtype)
        out = np.nextafter(x, y)
        self.inputs = {'x': x, 'y': y}
        self.outputs = {'out': out}

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output()

    def init_dtype(self):
        if False:
            i = 10
            return i + 15
        self.dtype = np.float64

class TestNextafterOPFP32(TestNextafterOP):

    def init_dtype(self):
        if False:
            print('Hello World!')
        self.dtype = np.float32
if __name__ == '__main__':
    unittest.main()