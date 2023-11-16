import unittest
import numpy as np
from op_test import OpTest
import paddle
from paddle import base, static
from paddle.pir_utils import test_with_pir_api
numpy_apis = {'real': np.real, 'imag': np.imag}
paddle_apis = {'real': paddle.real, 'imag': paddle.imag}

class TestRealOp(OpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        self.op_type = 'real'
        self.python_api = paddle.real
        self.dtype = np.float64
        self.init_input_output()
        self.init_grad_input_output()

    def init_input_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.inputs = {'X': np.random.random((20, 5)).astype(self.dtype) + 1j * np.random.random((20, 5)).astype(self.dtype)}
        self.outputs = {'Out': numpy_apis[self.op_type](self.inputs['X'])}

    def init_grad_input_output(self):
        if False:
            while True:
                i = 10
        self.grad_out = np.ones((20, 5), self.dtype)
        self.grad_x = np.real(self.grad_out) + 1j * np.zeros(self.grad_out.shape)

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output(check_pir=True)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad(['X'], 'Out', user_defined_grads=[self.grad_x], user_defined_grad_outputs=[self.grad_out], check_pir=True)

class TestImagOp(TestRealOp):

    def setUp(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        self.op_type = 'imag'
        self.python_api = paddle.imag
        self.dtype = np.float64
        self.init_input_output()
        self.init_grad_input_output()

    def init_grad_input_output(self):
        if False:
            return 10
        self.grad_out = np.ones((20, 5), self.dtype)
        self.grad_x = np.zeros(self.grad_out.shape) + 1j * np.real(self.grad_out)

class TestRealAPI(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        paddle.enable_static()
        self.api = 'real'
        self.dtypes = ['complex64', 'complex128']
        self.places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))
        self._shape = [2, 20, 2, 3]

    @test_with_pir_api
    def test_in_static_mode(self):
        if False:
            i = 10
            return i + 15

        def init_input_output(dtype):
            if False:
                i = 10
                return i + 15
            input = np.random.random(self._shape).astype(dtype) + 1j * np.random.random(self._shape).astype(dtype)
            return ({'x': input}, numpy_apis[self.api](input))
        for dtype in self.dtypes:
            (input_dict, np_res) = init_input_output(dtype)
            for place in self.places:
                with static.program_guard(static.Program()):
                    x = static.data(name='x', shape=self._shape, dtype=dtype)
                    out = paddle_apis[self.api](x)
                    exe = static.Executor(place)
                    out_value = exe.run(feed=input_dict, fetch_list=[out])
                    np.testing.assert_array_equal(np_res, out_value[0])

    def test_in_dynamic_mode(self):
        if False:
            while True:
                i = 10
        for dtype in self.dtypes:
            input = np.random.random(self._shape).astype(dtype) + 1j * np.random.random(self._shape).astype(dtype)
            np_res = numpy_apis[self.api](input)
            for place in self.places:
                with base.dygraph.guard(place):
                    input_t = paddle.to_tensor(input)
                    res = paddle_apis[self.api](input_t).numpy()
                    np.testing.assert_array_equal(np_res, res)
                    res_t = input_t.real().numpy() if self.api == 'real' else input_t.imag().numpy()
                    np.testing.assert_array_equal(np_res, res_t)

    def test_name_argument(self):
        if False:
            print('Hello World!')
        with static.program_guard(static.Program()):
            x = static.data(name='x', shape=self._shape, dtype=self.dtypes[0])
            out = paddle_apis[self.api](x, name='real_res')
            self.assertTrue('real_res' in out.name)

    def test_dtype_error(self):
        if False:
            return 10
        with self.assertRaises(TypeError):
            with static.program_guard(static.Program()):
                x = static.data(name='x', shape=self._shape, dtype='float32')
                out = paddle_apis[self.api](x, name='real_res')
        with self.assertRaises(RuntimeError):
            with base.dygraph.guard():
                input = np.random.random(self._shape).astype('float32')
                input_t = paddle.to_tensor(input)
                res = paddle_apis[self.api](input_t)

class TestImagAPI(TestRealAPI):

    def setUp(self):
        if False:
            return 10
        paddle.enable_static()
        self.api = 'imag'
        self.dtypes = ['complex64', 'complex128']
        self.places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))
        self._shape = [2, 20, 2, 3]
if __name__ == '__main__':
    unittest.main()