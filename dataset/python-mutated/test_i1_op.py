import unittest
import numpy as np
from op_test import OpTest
from scipy import special
import paddle
from paddle.base import core
from paddle.pir_utils import test_with_pir_api
np.random.seed(42)
paddle.seed(42)

def reference_i1(x):
    if False:
        i = 10
        return i + 15
    return special.i1(x)

def reference_i1_grad(x, dout):
    if False:
        print('Hello World!')
    eps = np.finfo(x.dtype).eps
    not_tiny = abs(x) > eps
    safe_x = np.where(not_tiny, x, eps)
    gradx = special.i0(safe_x) - special.i1(x) / safe_x
    gradx = np.where(not_tiny, gradx, 0.5)
    return dout * gradx

class Testi1API(unittest.TestCase):
    DTYPE = 'float64'
    DATA = [0, 1, 2, 3, 4, 5]

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.x = np.array(self.DATA).astype(self.DTYPE)
        self.place = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def test_api_static(self):
        if False:
            for i in range(10):
                print('nop')

        @test_with_pir_api
        def run(place):
            if False:
                print('Hello World!')
            paddle.enable_static()
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data(name='x', shape=self.x.shape, dtype=self.DTYPE)
                out = paddle.i1(x)
                exe = paddle.static.Executor(place)
                res = exe.run(paddle.static.default_main_program(), feed={'x': self.x}, fetch_list=[out])
                out_ref = reference_i1(self.x)
                np.testing.assert_allclose(res[0], out_ref, rtol=1e-05)
            paddle.disable_static()
        for place in self.place:
            run(place)

    def test_api_dygraph(self):
        if False:
            while True:
                i = 10

        def run(place):
            if False:
                for i in range(10):
                    print('nop')
            paddle.disable_static(place)
            x = paddle.to_tensor(self.x)
            out = paddle.i1(x)
            out_ref = reference_i1(self.x)
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=1e-05)
            paddle.enable_static()
        for place in self.place:
            run(place)

    def test_empty_input_error(self):
        if False:
            i = 10
            return i + 15
        for place in self.place:
            paddle.disable_static(place)
            x = None
            self.assertRaises(ValueError, paddle.i1, x)
            paddle.enable_static()

class Testi1Float32Zero2EightCase(Testi1API):
    DTYPE = 'float32'
    DATA = [0, 1, 2, 3, 4, 5, 6, 7, 8]

class Testi1Float32OverEightCase(Testi1API):
    DTYPE = 'float32'
    DATA = [9, 10, 11, 12, 13, 14, 15, 16, 17]

class Testi1Float64Zero2EightCase(Testi1API):
    DTYPE = 'float64'
    DATA = [0, 1, 2, 3, 4, 5, 6, 7, 8]

class Testi1Float64OverEightCase(Testi1API):
    DTYPE = 'float64'
    DATA = [9, 10, 11, 12, 13, 14, 15, 16, 17]

class TestI1Op(OpTest):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'i1'
        self.python_api = paddle.i1
        self.init_config()
        self.outputs = {'out': self.target}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            return 10
        self.check_grad(['x'], 'out', user_defined_grads=[reference_i1_grad(self.case, 1 / self.case.size)], check_pir=True)

    def init_config(self):
        if False:
            print('Hello World!')
        zero_case = np.zeros(1).astype('float64')
        rand_case = np.random.randn(250).astype('float64')
        over_eight_case = np.random.uniform(low=8, high=9, size=250).astype('float64')
        self.case = np.concatenate([zero_case, rand_case, over_eight_case])
        self.inputs = {'x': self.case}
        self.target = reference_i1(self.inputs['x'])
if __name__ == '__main__':
    unittest.main()