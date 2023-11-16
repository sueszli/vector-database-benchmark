import unittest
import numpy as np
import paddle

def ref_var(x, axis=None, unbiased=True, keepdim=False):
    if False:
        i = 10
        return i + 15
    ddof = 1 if unbiased else 0
    if isinstance(axis, int):
        axis = (axis,)
    if axis is not None:
        axis = tuple(axis)
    return np.var(x, axis=axis, ddof=ddof, keepdims=keepdim)

class TestVarAPI(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.dtype = 'float64'
        self.shape = [1, 3, 4, 10]
        self.axis = [1, 3]
        self.keepdim = False
        self.unbiased = True
        self.set_attrs()
        self.x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.place = paddle.CUDAPlace(0) if paddle.base.core.is_compiled_with_cuda() else paddle.CPUPlace()

    def set_attrs(self):
        if False:
            print('Hello World!')
        pass

    def static(self):
        if False:
            for i in range(10):
                print('nop')
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape, self.dtype)
            out = paddle.var(x, self.axis, self.unbiased, self.keepdim)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x}, fetch_list=[out])
        return res[0]

    def dygraph(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        out = paddle.var(x, self.axis, self.unbiased, self.keepdim)
        paddle.enable_static()
        return out.numpy()

    def test_api(self):
        if False:
            print('Hello World!')
        out_ref = ref_var(self.x, self.axis, self.unbiased, self.keepdim)
        out_dygraph = self.dygraph()
        out_static = self.static()
        for out in [out_dygraph, out_static]:
            np.testing.assert_allclose(out_ref, out, rtol=1e-05)
            self.assertTrue(np.equal(out_ref.shape, out.shape).all())

class TestVarAPI_dtype(TestVarAPI):

    def set_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.dtype = 'float32'

class TestVarAPI_axis_int(TestVarAPI):

    def set_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.axis = 2

class TestVarAPI_axis_list(TestVarAPI):

    def set_attrs(self):
        if False:
            i = 10
            return i + 15
        self.axis = [1, 2]

class TestVarAPI_axis_tuple(TestVarAPI):

    def set_attrs(self):
        if False:
            print('Hello World!')
        self.axis = (1, 3)

class TestVarAPI_keepdim(TestVarAPI):

    def set_attrs(self):
        if False:
            i = 10
            return i + 15
        self.keepdim = False

class TestVarAPI_unbiased(TestVarAPI):

    def set_attrs(self):
        if False:
            while True:
                i = 10
        self.unbiased = False

class TestVarAPI_alias(unittest.TestCase):

    def test_alias(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static()
        x = paddle.to_tensor(np.array([10, 12], 'float32'))
        out1 = paddle.var(x).numpy()
        out2 = paddle.tensor.var(x).numpy()
        out3 = paddle.tensor.stat.var(x).numpy()
        np.testing.assert_allclose(out1, out2, rtol=1e-05)
        np.testing.assert_allclose(out1, out3, rtol=1e-05)
        paddle.enable_static()

class TestVarError(unittest.TestCase):

    def test_error(self):
        if False:
            while True:
                i = 10
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', [2, 3, 4], 'int32')
            self.assertRaises(TypeError, paddle.var, x)
if __name__ == '__main__':
    unittest.main()