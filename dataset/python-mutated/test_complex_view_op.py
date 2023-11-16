import unittest
import numpy as np
from op_test import OpTest
import paddle
from paddle import static
from paddle.base import dygraph
from paddle.pir_utils import test_with_pir_api
paddle.enable_static()

def ref_view_as_complex(x):
    if False:
        print('Hello World!')
    (real, imag) = (np.take(x, 0, axis=-1), np.take(x, 1, axis=-1))
    return real + 1j * imag

def ref_view_as_real(x):
    if False:
        for i in range(10):
            print('nop')
    return np.stack([x.real, x.imag], -1)

class TestViewAsComplexOp(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'as_complex'
        self.python_api = paddle.as_complex
        x = np.random.randn(10, 10, 2).astype('float64')
        out_ref = ref_view_as_complex(x)
        self.inputs = {'X': x}
        self.outputs = {'Out': out_ref}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['X'], 'Out')

class TestViewAsRealOp(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'as_real'
        real = np.random.randn(10, 10).astype('float64')
        imag = np.random.randn(10, 10).astype('float64')
        x = real + 1j * imag
        out_ref = ref_view_as_real(x)
        self.inputs = {'X': x}
        self.outputs = {'Out': out_ref}
        self.python_api = paddle.as_real

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad(['X'], 'Out')

class TestViewAsComplexAPI(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.x = np.random.randn(10, 10, 2)
        self.out = ref_view_as_complex(self.x)

    def test_dygraph(self):
        if False:
            for i in range(10):
                print('nop')
        with dygraph.guard():
            x = paddle.to_tensor(self.x)
            out_np = paddle.as_complex(x).numpy()
        np.testing.assert_allclose(self.out, out_np, rtol=1e-05)

    @test_with_pir_api
    def test_static(self):
        if False:
            for i in range(10):
                print('nop')
        (mp, sp) = (static.Program(), static.Program())
        with static.program_guard(mp, sp):
            x = static.data('x', shape=[10, 10, 2], dtype='float64')
            out = paddle.as_complex(x)
        exe = static.Executor()
        exe.run(sp)
        [out_np] = exe.run(mp, feed={'x': self.x}, fetch_list=[out])
        np.testing.assert_allclose(self.out, out_np, rtol=1e-05)

class TestViewAsRealAPI(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.x = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)
        self.out = ref_view_as_real(self.x)

    def test_dygraph(self):
        if False:
            for i in range(10):
                print('nop')
        with dygraph.guard():
            x = paddle.to_tensor(self.x)
            out_np = paddle.as_real(x).numpy()
        np.testing.assert_allclose(self.out, out_np, rtol=1e-05)

    @test_with_pir_api
    def test_static(self):
        if False:
            for i in range(10):
                print('nop')
        (mp, sp) = (static.Program(), static.Program())
        with static.program_guard(mp, sp):
            x = static.data('x', shape=[10, 10], dtype='complex128')
            out = paddle.as_real(x)
        exe = static.Executor()
        exe.run(sp)
        [out_np] = exe.run(mp, feed={'x': self.x}, fetch_list=[out])
        np.testing.assert_allclose(self.out, out_np, rtol=1e-05)
if __name__ == '__main__':
    unittest.main()