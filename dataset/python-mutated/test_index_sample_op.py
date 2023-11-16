import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16
import paddle
from paddle import base
from paddle.base import core

class TestIndexSampleOp(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'index_sample'
        self.python_api = paddle.index_sample
        self.config()
        xnp = np.random.random(self.x_shape).astype(self.x_type)
        if self.x_type == np.complex64 or self.x_type == np.complex128:
            xnp = (np.random.random(self.x_shape) + 1j * np.random.random(self.x_shape)).astype(self.x_type)
        indexnp = np.random.randint(low=0, high=self.x_shape[1], size=self.index_shape).astype(self.index_type)
        self.inputs = {'X': xnp, 'Index': indexnp}
        index_array = []
        for i in range(self.index_shape[0]):
            for j in indexnp[i]:
                index_array.append(xnp[i, j])
        index_array = np.array(index_array).astype(self.x_type)
        out = np.reshape(index_array, self.index_shape)
        self.outputs = {'Out': out}

    def test_check_output(self):
        if False:
            return 10
        self.check_output()

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_grad(['X'], 'Out')

    def config(self):
        if False:
            while True:
                i = 10
        '\n        For multi-dimension input\n        '
        self.x_shape = (10, 20)
        self.x_type = 'float64'
        self.index_shape = (10, 10)
        self.index_type = 'int32'

class TestCase1(TestIndexSampleOp):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        For one dimension input\n        '
        self.x_shape = (100, 1)
        self.x_type = 'float64'
        self.index_shape = (100, 1)
        self.index_type = 'int32'

class TestCase2(TestIndexSampleOp):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        For int64_t index type\n        '
        self.x_shape = (10, 100)
        self.x_type = 'float64'
        self.index_shape = (10, 10)
        self.index_type = 'int64'

class TestCase3(TestIndexSampleOp):

    def config(self):
        if False:
            i = 10
            return i + 15
        '\n        For int index type\n        '
        self.x_shape = (10, 100)
        self.x_type = 'float64'
        self.index_shape = (10, 10)
        self.index_type = 'int32'

class TestCase4(TestIndexSampleOp):

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        For int64 index type\n        '
        self.x_shape = (10, 128)
        self.x_type = 'float64'
        self.index_shape = (10, 64)
        self.index_type = 'int64'

class TestCase5(TestIndexSampleOp):

    def config(self):
        if False:
            print('Hello World!')
        '\n        For float16 x type\n        '
        self.x_shape = (10, 128)
        self.x_type = 'float16'
        self.index_shape = (10, 64)
        self.index_type = 'int32'

class TestCase6(TestIndexSampleOp):

    def config(self):
        if False:
            while True:
                i = 10
        '\n        For float16 x type\n        '
        self.x_shape = (10, 128)
        self.x_type = 'float16'
        self.index_shape = (10, 64)
        self.index_type = 'int64'

class TestIndexSampleComplex64(TestIndexSampleOp):

    def config(self):
        if False:
            i = 10
            return i + 15
        '\n        For complex64 x type\n        '
        self.x_shape = (10, 128)
        self.x_type = np.complex64
        self.index_shape = (10, 64)
        self.index_type = 'int64'

class TestIndexSampleComplex128(TestIndexSampleOp):

    def config(self):
        if False:
            print('Hello World!')
        '\n        For complex64 x type\n        '
        self.x_shape = (10, 128)
        self.x_type = np.complex128
        self.index_shape = (10, 64)
        self.index_type = 'int64'

@unittest.skipIf(not core.is_compiled_with_cuda() or not core.is_bfloat16_supported(core.CUDAPlace(0)), 'core is not compiled with CUDA or not support bfloat16')
class TestIndexSampleBF16Op(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'index_sample'
        self.python_api = paddle.index_sample
        self.config()
        xnp = np.random.random(self.x_shape).astype(self.x_type)
        indexnp = np.random.randint(low=0, high=self.x_shape[1], size=self.index_shape).astype(self.index_type)
        self.inputs = {'X': xnp, 'Index': indexnp}
        index_array = []
        for i in range(self.index_shape[0]):
            for j in indexnp[i]:
                index_array.append(xnp[i, j])
        index_array = np.array(index_array).astype(self.x_type)
        out = np.reshape(index_array, self.index_shape)
        self.outputs = {'Out': out}
        self.inputs['X'] = convert_float_to_uint16(self.inputs['X'])
        self.outputs['Out'] = convert_float_to_uint16(self.outputs['Out'])
        self.place = core.CUDAPlace(0)

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        if False:
            print('Hello World!')
        self.check_grad_with_place(self.place, ['X'], 'Out')

    def config(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        For multi-dimension input\n        '
        self.x_shape = (10, 20)
        self.x_type = 'float32'
        self.dtype = np.uint16
        self.index_shape = (10, 10)
        self.index_type = 'int32'

class TestIndexSampleShape(unittest.TestCase):

    def test_shape(self):
        if False:
            while True:
                i = 10
        paddle.enable_static()
        x_shape = (2, 5)
        x_type = 'float64'
        x_np = np.random.random(x_shape).astype(x_type)
        index_shape = (2, 3)
        index_type = 'int32'
        index_np = np.random.randint(low=0, high=x_shape[1], size=index_shape).astype(index_type)
        x = paddle.static.data(name='x', shape=[-1, 5], dtype='float64')
        index = paddle.static.data(name='index', shape=[-1, 3], dtype='int32')
        output = paddle.index_sample(x=x, index=index)
        place = base.CPUPlace()
        exe = base.Executor(place=place)
        exe.run(base.default_startup_program())
        feed = {'x': x_np, 'index': index_np}
        res = exe.run(feed=feed, fetch_list=[output])

class TestIndexSampleDynamic(unittest.TestCase):

    def test_result(self):
        if False:
            for i in range(10):
                print('nop')
        with base.dygraph.guard():
            x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], dtype='float32')
            index = paddle.to_tensor([[0, 1, 2], [1, 2, 3], [0, 0, 0]], dtype='int32')
            out_z1 = paddle.index_sample(x, index)
            except_output = np.array([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0], [9.0, 9.0, 9.0]])
            assert out_z1.numpy().all() == except_output.all()
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()