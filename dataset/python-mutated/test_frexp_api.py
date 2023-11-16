import unittest
import numpy as np
import paddle
import paddle.base

class TestFrexpAPI(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        np.random.seed(1024)
        self.rtol = 1e-05
        self.atol = 1e-08
        self.place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
        self.set_input()

    def set_input(self):
        if False:
            while True:
                i = 10
        self.x_np = np.random.uniform(-3, 3, [10, 12]).astype('float32')

    def test_static_api(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            input_data = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
            out = paddle.frexp(input_data)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = np.frexp(self.x_np)
        for (n, p) in zip(out_ref, res):
            np.testing.assert_allclose(n, p, rtol=self.rtol, atol=self.atol)

    def test_dygraph_api(self):
        if False:
            print('Hello World!')
        paddle.disable_static(self.place)
        input_num = paddle.to_tensor(self.x_np)
        out1 = np.frexp(self.x_np)
        out2 = paddle.frexp(input_num)
        np.testing.assert_allclose(out1, out2, rtol=1e-05)
        out1 = np.frexp(self.x_np)
        out2 = input_num.frexp()
        np.testing.assert_allclose(out1, out2, rtol=1e-05)
        paddle.enable_static()

class TestSplitsFloat32Case1(TestFrexpAPI):
    """
    Test num_or_sections which is an integer and data type is float32.
    """

    def set_input(self):
        if False:
            print('Hello World!')
        self.x_np = np.random.uniform(-1, 1, [4, 5, 2]).astype('float32')

class TestSplitsFloat64Case1(TestFrexpAPI):
    """
    Test num_or_sections which is an integer and data type is float64.
    """

    def set_input(self):
        if False:
            return 10
        self.x_np = np.random.uniform(-3, 3, [10, 12]).astype('float64')

class TestSplitsFloat64Case2(TestFrexpAPI):
    """
    Test num_or_sections which is an integer and data type is float64.
    """

    def set_input(self):
        if False:
            while True:
                i = 10
        self.x_np = np.random.uniform(-1, 1, [4, 5, 2]).astype('float64')
if __name__ == '__main__':
    unittest.main()