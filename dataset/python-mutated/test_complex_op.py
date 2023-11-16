import unittest
import numpy as np
from op_test import OpTest
import paddle
from paddle import static
from paddle.base import dygraph
from paddle.pir_utils import test_with_pir_api
paddle.enable_static()

def ref_complex(x, y):
    if False:
        return 10
    return x + 1j * y

class TestComplexOp(OpTest):

    def init_spec(self):
        if False:
            print('Hello World!')
        self.x_shape = [10, 10]
        self.y_shape = [10, 10]
        self.dtype = 'float64'

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'complex'
        self.python_api = paddle.complex
        self.init_spec()
        x = np.random.randn(*self.x_shape).astype(self.dtype)
        y = np.random.randn(*self.y_shape).astype(self.dtype)
        out_ref = ref_complex(x, y)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out_ref}

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad(['X', 'Y'], 'Out', check_pir=True)

    def test_check_grad_ignore_x(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['Y'], 'Out', no_grad_set=set('X'), check_pir=True)

    def test_check_grad_ignore_y(self):
        if False:
            return 10
        self.check_grad(['X'], 'Out', no_grad_set=set('Y'), check_pir=True)

class TestComplexOpBroadcast1(TestComplexOp):

    def init_spec(self):
        if False:
            return 10
        self.x_shape = [10, 3, 1, 4]
        self.y_shape = [100, 1]
        self.dtype = 'float64'

class TestComplexOpBroadcast2(TestComplexOp):

    def init_spec(self):
        if False:
            for i in range(10):
                print('nop')
        self.x_shape = [100, 1]
        self.y_shape = [10, 3, 1, 4]
        self.dtype = 'float32'

class TestComplexOpBroadcast3(TestComplexOp):

    def init_spec(self):
        if False:
            return 10
        self.x_shape = [1, 100]
        self.y_shape = [100]
        self.dtype = 'float32'

class TestComplexAPI(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.x = np.random.randn(10, 10)
        self.y = np.random.randn(10, 10)
        self.out = ref_complex(self.x, self.y)

    def test_dygraph(self):
        if False:
            i = 10
            return i + 15
        with dygraph.guard():
            x = paddle.to_tensor(self.x)
            y = paddle.to_tensor(self.y)
            out_np = paddle.complex(x, y).numpy()
        np.testing.assert_allclose(self.out, out_np, rtol=1e-05)

    @test_with_pir_api
    def test_static(self):
        if False:
            for i in range(10):
                print('nop')
        (mp, sp) = (static.Program(), static.Program())
        with static.program_guard(mp, sp):
            x = static.data('x', shape=[10, 10], dtype='float64')
            y = static.data('y', shape=[10, 10], dtype='float64')
            out = paddle.complex(x, y)
        exe = static.Executor()
        exe.run(sp)
        [out_np] = exe.run(mp, feed={'x': self.x, 'y': self.y}, fetch_list=[out])
        np.testing.assert_allclose(self.out, out_np, rtol=1e-05)
if __name__ == '__main__':
    unittest.main()