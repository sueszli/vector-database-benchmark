import unittest
import numpy as np
import paddle

class TestNegOp(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.init_dtype_type()
        self.input = (np.random.random((32, 8)) * 100).astype(self.dtype)

    def init_dtype_type(self):
        if False:
            return 10
        self.dtype = np.float64

    def run_imperative(self):
        if False:
            while True:
                i = 10
        input = paddle.to_tensor(self.input)
        dy_result = paddle.neg(input)
        expected_result = np.negative(self.input)
        np.testing.assert_allclose(dy_result.numpy(), expected_result, rtol=1e-05)

    def run_static(self, use_gpu=False):
        if False:
            i = 10
            return i + 15
        input = paddle.static.data(name='input', shape=[32, 8], dtype=self.dtype)
        result = paddle.neg(input)
        place = paddle.CUDAPlace(0) if use_gpu else paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())
        st_result = exe.run(feed={'input': self.input}, fetch_list=[result])
        expected_result = np.negative(self.input)
        np.testing.assert_allclose(st_result[0], expected_result, rtol=1e-05)

    def test_cpu(self):
        if False:
            return 10
        paddle.disable_static(place=paddle.CPUPlace())
        self.run_imperative()
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            self.run_static()

    def test_gpu(self):
        if False:
            return 10
        if not paddle.base.core.is_compiled_with_cuda():
            return
        paddle.disable_static(place=paddle.CUDAPlace(0))
        self.run_imperative()
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            self.run_static(use_gpu=True)

class TestNegOpFp32(TestNegOp):

    def init_dtype_type(self):
        if False:
            print('Hello World!')
        self.dtype = np.float32

class TestNegOpInt64(TestNegOp):

    def init_dtype_type(self):
        if False:
            return 10
        self.dtype = np.int64

class TestNegOpInt32(TestNegOp):

    def init_dtype_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.int32

class TestNegOpInt16(TestNegOp):

    def init_dtype_type(self):
        if False:
            print('Hello World!')
        self.dtype = np.int16

class TestNegOpInt8(TestNegOp):

    def init_dtype_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.int8
if __name__ == '__main__':
    unittest.main()