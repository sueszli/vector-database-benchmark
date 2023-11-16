import unittest
import numpy as np
from op_test import OpTest
import paddle
from paddle import base
from paddle.base import core
from paddle.pir_utils import test_with_pir_api

class TestInverseOp(OpTest):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.matrix_shape = [10, 10]
        self.dtype = 'float64'
        self.python_api = paddle.tensor.math.inverse

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'inverse'
        self.config()
        np.random.seed(123)
        mat = np.random.random(self.matrix_shape).astype(self.dtype)
        inverse = np.linalg.inv(mat)
        self.inputs = {'Input': mat}
        self.outputs = {'Output': inverse}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_pir=True)

    def test_grad(self):
        if False:
            print('Hello World!')
        self.check_grad(['Input'], 'Output', check_pir=True)

class TestInverseOpBatched(TestInverseOp):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.matrix_shape = [8, 4, 4]
        self.dtype = 'float64'
        self.python_api = paddle.tensor.math.inverse

class TestInverseOpLarge(TestInverseOp):

    def config(self):
        if False:
            i = 10
            return i + 15
        self.matrix_shape = [32, 32]
        self.dtype = 'float64'
        self.python_api = paddle.tensor.math.inverse

    def test_grad(self):
        if False:
            return 10
        self.check_grad(['Input'], 'Output', max_relative_error=1e-06, check_pir=True)

class TestInverseOpFP32(TestInverseOp):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.matrix_shape = [10, 10]
        self.dtype = 'float32'
        self.python_api = paddle.tensor.math.inverse

    def test_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['Input'], 'Output', max_relative_error=0.01, check_pir=True)

class TestInverseOpBatchedFP32(TestInverseOpFP32):

    def config(self):
        if False:
            return 10
        self.matrix_shape = [8, 4, 4]
        self.dtype = 'float32'
        self.python_api = paddle.tensor.math.inverse

class TestInverseOpLargeFP32(TestInverseOpFP32):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        self.matrix_shape = [32, 32]
        self.dtype = 'float32'
        self.python_api = paddle.tensor.math.inverse

class TestInverseAPI(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        np.random.seed(123)
        self.places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def check_static_result(self, place):
        if False:
            i = 10
            return i + 15
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            input = paddle.static.data(name='input', shape=[4, 4], dtype='float64')
            result = paddle.inverse(x=input)
            input_np = np.random.random([4, 4]).astype('float64')
            result_np = np.linalg.inv(input_np)
            exe = base.Executor(place)
            fetches = exe.run(paddle.static.default_main_program(), feed={'input': input_np}, fetch_list=[result])
            np.testing.assert_allclose(fetches[0], np.linalg.inv(input_np), rtol=1e-05)

    @test_with_pir_api
    def test_static(self):
        if False:
            print('Hello World!')
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        if False:
            print('Hello World!')
        for place in self.places:
            with base.dygraph.guard(place):
                input_np = np.random.random([4, 4]).astype('float64')
                input = base.dygraph.to_variable(input_np)
                result = paddle.inverse(input)
                np.testing.assert_allclose(result.numpy(), np.linalg.inv(input_np), rtol=1e-05)

class TestInverseAPIError(unittest.TestCase):

    def test_errors(self):
        if False:
            i = 10
            return i + 15
        input_np = np.random.random([4, 4]).astype('float64')
        self.assertRaises(TypeError, paddle.inverse, input_np)
        for dtype in ['bool', 'int32', 'int64', 'float16']:
            input = paddle.static.data(name='input_' + dtype, shape=[4, 4], dtype=dtype)
            self.assertRaises(TypeError, paddle.inverse, input)
        input = paddle.static.data(name='input_1', shape=[4, 4], dtype='float32')
        out = paddle.static.data(name='output', shape=[4, 4], dtype='float64')
        self.assertRaises(TypeError, paddle.inverse, input, out)
        input = paddle.static.data(name='input_2', shape=[4], dtype='float32')
        self.assertRaises(ValueError, paddle.inverse, input)

class TestInverseSingularAPI(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def check_static_result(self, place):
        if False:
            return 10
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            input = paddle.static.data(name='input', shape=[4, 4], dtype='float64')
            result = paddle.inverse(x=input)
            input_np = np.zeros([4, 4]).astype('float64')
            exe = base.Executor(place)
            try:
                fetches = exe.run(paddle.static.default_main_program(), feed={'input': input_np}, fetch_list=[result])
            except RuntimeError as ex:
                print('The mat is singular')
            except ValueError as ex:
                print('The mat is singular')

    @test_with_pir_api
    def test_static(self):
        if False:
            return 10
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        if False:
            i = 10
            return i + 15
        for place in self.places:
            with base.dygraph.guard(place):
                input_np = np.ones([4, 4]).astype('float64')
                input = base.dygraph.to_variable(input_np)
                try:
                    result = paddle.inverse(input)
                except RuntimeError as ex:
                    print('The mat is singular')
                except ValueError as ex:
                    print('The mat is singular')
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()