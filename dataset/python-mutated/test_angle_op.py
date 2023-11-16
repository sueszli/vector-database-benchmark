import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16
import paddle
from paddle import static
from paddle.base import core, dygraph
from paddle.pir_utils import test_with_pir_api
paddle.enable_static()

def angle_grad(x, dout):
    if False:
        i = 10
        return i + 15
    if np.iscomplexobj(x):

        def angle_grad_element(xi, douti):
            if False:
                print('Hello World!')
            if xi == 0:
                return 0
            rsquare = np.abs(xi) ** 2
            return -douti * xi.imag / rsquare + 1j * douti * xi.real / rsquare
        return np.vectorize(angle_grad_element)(x, dout)
    else:
        return np.zeros_like(x).astype(x.dtype)

class TestAngleOpFloat(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.op_type = 'angle'
        self.python_api = paddle.angle
        self.dtype = 'float64'
        self.x = np.linspace(-5, 5, 101).astype(self.dtype)
        out_ref = np.angle(self.x)
        self.inputs = {'X': self.x}
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
        self.check_grad(['X'], 'Out', user_defined_grads=[angle_grad(self.x, np.ones_like(self.x) / self.x.size)], check_pir=True)

class TestAngleFP16Op(TestAngleOpFloat):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'angle'
        self.python_api = paddle.angle
        self.dtype = 'float16'
        self.x = np.linspace(-5, 5, 101).astype(self.dtype)
        out_ref = np.angle(self.x)
        self.inputs = {'X': self.x}
        self.outputs = {'Out': out_ref}

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not compiled with CUDA or not support bfloat16')
class TestAngleBF16Op(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'angle'
        self.python_api = paddle.angle
        self.dtype = np.uint16
        self.np_dtype = np.float32
        self.x = np.linspace(-5, 5, 101).astype(self.np_dtype)
        out_ref = np.angle(self.x)
        self.inputs = {'X': self.x}
        self.outputs = {'Out': out_ref}
        self.inputs['X'] = convert_float_to_uint16(self.inputs['X'])
        self.outputs['Out'] = convert_float_to_uint16(self.outputs['Out'])
        self.place = core.CUDAPlace(0)

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output_with_place(self.place, check_pir=True)

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        self.check_grad_with_place(self.place, ['X'], 'Out', user_defined_grads=[angle_grad(self.x, np.ones_like(self.x) / self.x.size)], check_pir=True)

class TestAngleOpComplex(OpTest):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'angle'
        self.python_api = paddle.angle
        self.dtype = 'complex128'
        real = np.expand_dims(np.linspace(-2, 2, 11), -1).astype('float64')
        imag = np.linspace(-2, 2, 11).astype('float64')
        self.x = real + 1j * imag
        out_ref = np.angle(self.x)
        self.inputs = {'X': self.x}
        self.outputs = {'Out': out_ref}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            return 10
        self.check_grad(['X'], 'Out', user_defined_grads=[angle_grad(self.x, np.ones_like(self.x) / self.x.size)], check_pir=True)

class TestAngleAPI(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.x = np.random.randn(2, 3) + 1j * np.random.randn(2, 3)
        self.out = np.angle(self.x)

    def test_dygraph(self):
        if False:
            i = 10
            return i + 15
        with dygraph.guard():
            x = paddle.to_tensor(self.x)
            out_np = paddle.angle(x).numpy()
        np.testing.assert_allclose(self.out, out_np, rtol=1e-05)

    @test_with_pir_api
    def test_static(self):
        if False:
            for i in range(10):
                print('nop')
        (mp, sp) = (static.Program(), static.Program())
        with static.program_guard(mp, sp):
            x = static.data('x', shape=[2, 3], dtype='complex128')
            out = paddle.angle(x)
        exe = static.Executor()
        exe.run(sp)
        [out_np] = exe.run(mp, feed={'x': self.x}, fetch_list=[out])
        np.testing.assert_allclose(self.out, out_np, rtol=1e-05)
if __name__ == '__main__':
    unittest.main()