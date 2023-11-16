import unittest
import numpy as np
import paddle
from paddle import base
from paddle.base import core
from paddle.pir_utils import test_with_pir_api

def ref_frac(x):
    if False:
        print('Hello World!')
    return x - np.trunc(x)

class TestFracAPI(unittest.TestCase):
    """Test Frac API"""

    def set_dtype(self):
        if False:
            return 10
        self.dtype = 'float64'

    def setUp(self):
        if False:
            return 10
        self.set_dtype()
        self.x_np = np.random.uniform(-3, 3, [2, 3]).astype(self.dtype)
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() else paddle.CPUPlace()

    @test_with_pir_api
    def test_api_static(self):
        if False:
            return 10
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            input = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
            out = paddle.frac(input)
            exe = base.Executor(self.place)
            (res,) = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = ref_frac(self.x_np)
        np.testing.assert_allclose(out_ref, res, rtol=1e-05)

    def test_api_dygraph(self):
        if False:
            return 10
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out = paddle.frac(x)
        out_ref = ref_frac(self.x_np)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)

    def test_api_eager(self):
        if False:
            while True:
                i = 10
        paddle.disable_static(self.place)
        x_tensor = paddle.to_tensor(self.x_np)
        out = paddle.frac(x_tensor)
        out_ref = ref_frac(self.x_np)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-05)
        paddle.enable_static()

class TestFracInt32(TestFracAPI):
    """Test Frac API with data type int32"""

    def set_dtype(self):
        if False:
            i = 10
            return i + 15
        self.dtype = 'int32'

class TestFracInt64(TestFracAPI):
    """Test Frac API with data type int64"""

    def set_dtype(self):
        if False:
            i = 10
            return i + 15
        self.dtype = 'int64'

class TestFracFloat32(TestFracAPI):
    """Test Frac API with data type float32"""

    def set_dtype(self):
        if False:
            i = 10
            return i + 15
        self.dtype = 'float32'

class TestFracError(unittest.TestCase):
    """Test Frac Error"""

    def setUp(self):
        if False:
            while True:
                i = 10
        self.x_np = np.random.uniform(-3, 3, [2, 3]).astype('int16')
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() else paddle.CPUPlace()

    @test_with_pir_api
    def test_static_error(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', [5, 5], 'bool')
            self.assertRaises(TypeError, paddle.frac, x)

    def test_dygraph_error(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np, dtype='int16')
        self.assertRaises(TypeError, paddle.frac, x)
if __name__ == '__main__':
    unittest.main()