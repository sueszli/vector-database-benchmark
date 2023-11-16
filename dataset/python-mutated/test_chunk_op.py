import unittest
import numpy as np
import paddle
from paddle import base
from paddle.base import Program, program_guard

class TestChunkOpError(unittest.TestCase):

    def test_errors(self):
        if False:
            return 10
        with program_guard(Program(), Program()):

            def test_axis_type():
                if False:
                    return 10
                x1 = paddle.static.data(shape=[4], dtype='float16', name='x3')
                paddle.chunk(x=x1, chunks=2, axis=3.2)
            self.assertRaises(TypeError, test_axis_type)

            def test_axis_variable_type():
                if False:
                    while True:
                        i = 10
                x2 = paddle.static.data(shape=[4], dtype='float16', name='x9')
                x3 = paddle.static.data(shape=[1], dtype='float16', name='x10')
                paddle.chunk(input=x2, chunks=2, axis=x3)
            self.assertRaises(TypeError, test_axis_variable_type)

            def test_chunks_type():
                if False:
                    return 10
                x4 = paddle.static.data(shape=[4], dtype='float16', name='x4')
                paddle.chunk(input=x4, chunks=2.1, axis=3)
            self.assertRaises(TypeError, test_chunks_type)

            def test_axis_type_tensor():
                if False:
                    i = 10
                    return i + 15
                x5 = paddle.static.data(shape=[4], dtype='float16', name='x6')
                paddle.chunk(input=x5, chunks=2, axis=3.2)
            self.assertRaises(TypeError, test_axis_type_tensor)
        with paddle.base.dygraph.guard():

            def test_0_chunks_tensor():
                if False:
                    return 10
                x = paddle.uniform([1, 1, 1], dtype='float32')
                paddle.chunk(x, chunks=0)
            self.assertRaises(ValueError, test_0_chunks_tensor)

class API_TestChunk(unittest.TestCase):

    def test_out(self):
        if False:
            i = 10
            return i + 15
        with base.program_guard(base.Program(), base.Program()):
            data1 = paddle.static.data('data1', shape=[4, 6, 6], dtype='float64')
            data2 = paddle.static.data('data2', shape=[1], dtype='int32')
            (x0, x1, x2) = paddle.chunk(data1, chunks=3, axis=data2)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            input1 = np.random.random([4, 6, 6]).astype('float64')
            input2 = np.array([2]).astype('int32')
            (r0, r1, r2) = exe.run(feed={'data1': input1, 'data2': input2}, fetch_list=[x0, x1, x2])
            (ex_x0, ex_x1, ex_x2) = np.array_split(input1, 3, axis=2)
            np.testing.assert_allclose(ex_x0, r0, rtol=1e-05)
            np.testing.assert_allclose(ex_x1, r1, rtol=1e-05)
            np.testing.assert_allclose(ex_x2, r2, rtol=1e-05)

class API_TestChunk1(unittest.TestCase):

    def test_out(self):
        if False:
            while True:
                i = 10
        with base.program_guard(base.Program(), base.Program()):
            data1 = paddle.static.data('data1', shape=[4, 6, 6], dtype='float64')
            (x0, x1, x2) = paddle.chunk(data1, chunks=3, axis=2)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            input1 = np.random.random([4, 6, 6]).astype('float64')
            (r0, r1, r2) = exe.run(feed={'data1': input1}, fetch_list=[x0, x1, x2])
            (ex_x0, ex_x1, ex_x2) = np.array_split(input1, 3, axis=2)
            np.testing.assert_allclose(ex_x0, r0, rtol=1e-05)
            np.testing.assert_allclose(ex_x1, r1, rtol=1e-05)
            np.testing.assert_allclose(ex_x2, r2, rtol=1e-05)

class API_TestDygraphChunk(unittest.TestCase):

    def test_out1(self):
        if False:
            print('Hello World!')
        with base.dygraph.guard():
            input_1 = np.random.random([4, 6, 6]).astype('int32')
            input = base.dygraph.to_variable(input_1)
            (x0, x1, x2) = paddle.chunk(input, chunks=3, axis=1)
            x0_out = x0.numpy()
            x1_out = x1.numpy()
            x2_out = x2.numpy()
            (ex_x0, ex_x1, ex_x2) = np.array_split(input_1, 3, axis=1)
        np.testing.assert_allclose(ex_x0, x0_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x1, x1_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x2, x2_out, rtol=1e-05)

    def test_out2(self):
        if False:
            i = 10
            return i + 15
        with base.dygraph.guard():
            input_1 = np.random.random([4, 6, 6]).astype('bool')
            input = base.dygraph.to_variable(input_1)
            (x0, x1, x2) = paddle.chunk(input, chunks=3, axis=1)
            x0_out = x0.numpy()
            x1_out = x1.numpy()
            x2_out = x2.numpy()
            (ex_x0, ex_x1, ex_x2) = np.array_split(input_1, 3, axis=1)
        np.testing.assert_allclose(ex_x0, x0_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x1, x1_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x2, x2_out, rtol=1e-05)

    def test_axis_tensor_input(self):
        if False:
            print('Hello World!')
        with base.dygraph.guard():
            input_1 = np.random.random([4, 6, 6]).astype('int32')
            input = base.dygraph.to_variable(input_1)
            num1 = paddle.full(shape=[1], fill_value=1, dtype='int32')
            (x0, x1, x2) = paddle.chunk(input, chunks=3, axis=num1)
            x0_out = x0.numpy()
            x1_out = x1.numpy()
            x2_out = x2.numpy()
            (ex_x0, ex_x1, ex_x2) = np.array_split(input_1, 3, axis=1)
        np.testing.assert_allclose(ex_x0, x0_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x1, x1_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x2, x2_out, rtol=1e-05)
if __name__ == '__main__':
    unittest.main()