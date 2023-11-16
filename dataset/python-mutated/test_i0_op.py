import unittest
import numpy as np
from op_test import OpTest
from scipy import special
import paddle
from paddle.base import core
from paddle.pir_utils import test_with_pir_api
np.random.seed(100)
paddle.seed(100)

def output_i0(x):
    if False:
        print('Hello World!')
    return special.i0(x)

def ref_i0_grad(x, dout):
    if False:
        while True:
            i = 10
    gradx = special.i1(x)
    return dout * gradx

class TestI0API(unittest.TestCase):
    DTYPE = 'float64'
    DATA = [0, 1, 2, 3, 4, 5]

    def setUp(self):
        if False:
            print('Hello World!')
        self.x = np.array(self.DATA).astype(self.DTYPE)
        self.out_ref = output_i0(self.x)
        self.place = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    @test_with_pir_api
    def test_api_static(self):
        if False:
            return 10

        def run(place):
            if False:
                return 10
            paddle.enable_static()
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data(name='x', shape=self.x.shape, dtype=self.DTYPE)
                out = paddle.i0(x)
                exe = paddle.static.Executor(place)
                res = exe.run(paddle.static.default_main_program(), feed={'x': self.x}, fetch_list=[out])
                np.testing.assert_allclose(res[0], self.out_ref, rtol=1e-05)
            paddle.disable_static()
        for place in self.place:
            run(place)

    def test_api_dygraph(self):
        if False:
            return 10

        def run(place):
            if False:
                print('Hello World!')
            paddle.disable_static(place)
            x = paddle.to_tensor(self.x)
            out = paddle.i0(x)
            out_ref = output_i0(self.x)
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=1e-05)
            paddle.enable_static()
        for place in self.place:
            run(place)

    def test_empty_input_error(self):
        if False:
            print('Hello World!')
        for place in self.place:
            paddle.disable_static(place)
            x = None
            self.assertRaises(ValueError, paddle.i0, x)
            paddle.enable_static()

class TestI0Float32Zero2EightCase(TestI0API):
    DTYPE = 'float32'
    DATA = [0, 1, 2, 3, 4, 5, 6, 7, 8]

class TestI0Float32OverEightCase(TestI0API):
    DTYPE = 'float32'
    DATA = [9, 10, 11, 12]

class TestI0Float64Zero2EightCase(TestI0API):
    DTYPE = 'float64'
    DATA = [0, 1, 2, 3, 4, 5, 6, 7, 8]

class TestI0Float64OverEightCase(TestI0API):
    DTYPE = 'float64'
    DATA = [9, 10, 11, 12]

class TestI0Op(OpTest):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'i0'
        self.python_api = paddle.i0
        self.init_config()
        self.outputs = {'out': self.target}

    def init_config(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = np.float64
        zero_case = np.zeros(1).astype(self.dtype)
        rand_case = np.random.randn(100).astype(self.dtype)
        one2eight_case = np.random.uniform(low=1, high=8, size=100).astype(self.dtype)
        over_eight_case = np.random.uniform(low=9, high=15, size=100).astype(self.dtype)
        self.case = np.concatenate([zero_case, rand_case, one2eight_case, over_eight_case])
        self.inputs = {'x': self.case}
        self.target = output_i0(self.inputs['x'])

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad(['x'], 'out', user_defined_grads=[ref_i0_grad(self.case, 1 / self.case.size)], check_pir=True)
if __name__ == '__main__':
    unittest.main()