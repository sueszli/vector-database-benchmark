import unittest
import numpy as np
import paddle
from paddle import base
from paddle.base import core
from paddle.pir_utils import test_with_pir_api
paddle.enable_static()

class TestMaxMinAmaxAminAPI(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.init_case()
        self.cal_np_out_and_gradient()
        self.place = base.CUDAPlace(0) if core.is_compiled_with_cuda() else base.CPUPlace()

    def init_case(self):
        if False:
            return 10
        self.x_np = np.array([[0.2, 0.3, 0.5, 0.9], [0.1, 0.2, 0.6, 0.7]])
        self.shape = [2, 4]
        self.dtype = 'float64'
        self.axis = 0
        self.keepdim = False

    def cal_np_out_and_gradient(self):
        if False:
            for i in range(10):
                print('nop')

        def _cal_np_out_and_gradient(func):
            if False:
                for i in range(10):
                    print('nop')
            if func == 'amax':
                out = np.amax(self.x_np, axis=self.axis, keepdims=self.keepdim)
            elif func == 'amin':
                out = np.amin(self.x_np, axis=self.axis, keepdims=self.keepdim)
            elif func == 'max':
                out = np.max(self.x_np, axis=self.axis, keepdims=self.keepdim)
            elif func == 'min':
                out = np.min(self.x_np, axis=self.axis, keepdims=self.keepdim)
            else:
                print('This unittest only test amax/amin/max/min, but now is', func)
            self.np_out[func] = out
            grad = np.zeros(self.shape)
            out_b = np.broadcast_to(out.view(), self.shape)
            grad[self.x_np == out_b] = 1
            if func in ['amax', 'amin']:
                grad_sum = grad.sum(self.axis).reshape(out.shape)
                grad_b = np.broadcast_to(grad_sum, self.shape)
                grad /= grad_sum
            self.np_grad[func] = grad
        self.np_out = {}
        self.np_grad = {}
        _cal_np_out_and_gradient('amax')
        _cal_np_out_and_gradient('amin')
        _cal_np_out_and_gradient('max')
        _cal_np_out_and_gradient('min')

    def _choose_paddle_func(self, func, x):
        if False:
            for i in range(10):
                print('nop')
        if func == 'amax':
            out = paddle.amax(x, self.axis, self.keepdim)
        elif func == 'amin':
            out = paddle.amin(x, self.axis, self.keepdim)
        elif func == 'max':
            out = paddle.max(x, self.axis, self.keepdim)
        elif func == 'min':
            out = paddle.min(x, self.axis, self.keepdim)
        else:
            print('This unittest only test amax/amin/max/min, but now is', func)
        return out

    @test_with_pir_api
    def test_static_graph(self):
        if False:
            for i in range(10):
                print('nop')

        def _test_static_graph(func):
            if False:
                return 10
            startup_program = base.Program()
            train_program = base.Program()
            with base.program_guard(startup_program, train_program):
                x = paddle.static.data(name='input', dtype=self.dtype, shape=self.shape)
                x.stop_gradient = False
                out = self._choose_paddle_func(func, x)
                exe = base.Executor(self.place)
                res = exe.run(feed={'input': self.x_np}, fetch_list=[out])
                self.assertTrue((np.array(res[0]) == self.np_out[func]).all())
        _test_static_graph('amax')
        _test_static_graph('amin')
        _test_static_graph('max')
        _test_static_graph('min')

    def test_dygraph(self):
        if False:
            while True:
                i = 10

        def _test_dygraph(func):
            if False:
                while True:
                    i = 10
            paddle.disable_static()
            x = paddle.to_tensor(self.x_np, dtype=self.dtype, stop_gradient=False)
            out = self._choose_paddle_func(func, x)
            grad_tensor = paddle.ones_like(x)
            paddle.autograd.backward([out], [grad_tensor], True)
            np.testing.assert_allclose(self.np_out[func], out.numpy(), rtol=1e-05)
            np.testing.assert_allclose(self.np_grad[func], x.grad, rtol=1e-05)
            paddle.enable_static()
        _test_dygraph('amax')
        _test_dygraph('amin')
        _test_dygraph('max')
        _test_dygraph('min')

class TestMaxMinAmaxAminAPI_ZeroDim(TestMaxMinAmaxAminAPI):

    def init_case(self):
        if False:
            while True:
                i = 10
        self.x_np = np.array(0.5)
        self.shape = []
        self.dtype = 'float64'
        self.axis = None
        self.keepdim = False

class TestMaxMinAmaxAminAPI2(TestMaxMinAmaxAminAPI):

    def init_case(self):
        if False:
            print('Hello World!')
        self.x_np = np.array([[0.2, 0.3, 0.9, 0.9], [0.1, 0.1, 0.6, 0.7]])
        self.shape = [2, 4]
        self.dtype = 'float64'
        self.axis = None
        self.keepdim = False

class TestMaxMinAmaxAminAPI3(TestMaxMinAmaxAminAPI):

    def init_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.x_np = np.array([[0.2, 0.3, 0.9, 0.9], [0.1, 0.1, 0.6, 0.7]])
        self.shape = [2, 4]
        self.dtype = 'float64'
        self.axis = 0
        self.keepdim = False

class TestMaxMinAmaxAminAPI4(TestMaxMinAmaxAminAPI):

    def init_case(self):
        if False:
            while True:
                i = 10
        self.x_np = np.array([[0.2, 0.3, 0.9, 0.9], [0.1, 0.1, 0.6, 0.7]])
        self.shape = [2, 4]
        self.dtype = 'float64'
        self.axis = 1
        self.keepdim = True

class TestMaxMinAmaxAminAPI5(TestMaxMinAmaxAminAPI):

    def init_case(self):
        if False:
            while True:
                i = 10
        self.x_np = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.int32)
        self.shape = [2, 2, 2]
        self.dtype = 'int32'
        self.axis = (0, 1)
        self.keepdim = False

class TestMaxMinAmaxAminAPI6(TestMaxMinAmaxAminAPI):

    def init_case(self):
        if False:
            for i in range(10):
                print('nop')
        self.x_np = np.array([[0.2, 0.9, 0.9, 0.9], [0.9, 0.9, 0.2, 0.2]])
        self.shape = [2, 4]
        self.dtype = 'float64'
        self.axis = None
        self.keepdim = False

class TestMaxMinAmaxAminAPI7(TestMaxMinAmaxAminAPI):

    def init_case(self):
        if False:
            while True:
                i = 10
        self.x_np = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.int32)
        self.shape = [2, 2, 2]
        self.dtype = 'int32'
        self.axis = (0, 1)
        self.keepdim = False

    def test_dygraph(self):
        if False:
            return 10

        def _test_dygraph(func):
            if False:
                return 10
            paddle.disable_static()
            x = paddle.to_tensor(self.x_np, dtype=self.dtype, stop_gradient=False)
            out = self._choose_paddle_func(func, x)
            loss = out * 2
            grad_tensor = paddle.ones_like(x)
            paddle.autograd.backward([loss], [grad_tensor], True)
            np.testing.assert_allclose(self.np_out[func], out.numpy(), rtol=1e-05)
            np.testing.assert_allclose(self.np_grad[func] * 2, x.grad, rtol=1e-05)
            paddle.enable_static()
        _test_dygraph('amax')
        _test_dygraph('amin')
        _test_dygraph('max')
        _test_dygraph('min')
if __name__ == '__main__':
    unittest.main()