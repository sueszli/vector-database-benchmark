import unittest
import numpy as np
import paddle
from paddle import base
from paddle.base import core
from paddle.pir_utils import test_with_pir_api

class TestSortOnCPU(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.place = core.CPUPlace()

    @test_with_pir_api
    def test_api_0(self):
        if False:
            print('Hello World!')
        with base.program_guard(base.Program()):
            input = paddle.static.data(name='input', shape=[2, 3, 4], dtype='float32')
            output = paddle.sort(x=input)
            exe = base.Executor(self.place)
            data = np.array([[[5, 8, 9, 5], [0, 0, 1, 7], [6, 9, 2, 4]], [[5, 2, 4, 2], [4, 7, 7, 9], [1, 7, 0, 6]]], dtype='float32')
            (result,) = exe.run(feed={'input': data}, fetch_list=[output])
            np_result = np.sort(result)
            self.assertEqual((result == np_result).all(), True)

    @test_with_pir_api
    def test_api_1(self):
        if False:
            while True:
                i = 10
        with base.program_guard(base.Program()):
            input = paddle.static.data(name='input', shape=[2, 3, 4], dtype='float32')
            output = paddle.sort(x=input, axis=1)
            exe = base.Executor(self.place)
            data = np.array([[[5, 8, 9, 5], [0, 0, 1, 7], [6, 9, 2, 4]], [[5, 2, 4, 2], [4, 7, 7, 9], [1, 7, 0, 6]]], dtype='float32')
            (result,) = exe.run(feed={'input': data}, fetch_list=[output])
            np_result = np.sort(result, axis=1)
            self.assertEqual((result == np_result).all(), True)

class TestSortOnGPU(TestSortOnCPU):

    def init_place(self):
        if False:
            return 10
        if core.is_compiled_with_cuda():
            self.place = core.CUDAPlace(0)
        else:
            self.place = core.CPUPlace()

class TestSortDygraph(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.input_data = np.random.rand(10, 10)
        if core.is_compiled_with_cuda():
            self.place = core.CUDAPlace(0)
        else:
            self.place = core.CPUPlace()

    def test_api_0(self):
        if False:
            print('Hello World!')
        paddle.disable_static(self.place)
        var_x = paddle.to_tensor(self.input_data)
        out = paddle.sort(var_x)
        self.assertEqual((np.sort(self.input_data) == out.numpy()).all(), True)
        paddle.enable_static()

    def test_api_1(self):
        if False:
            while True:
                i = 10
        paddle.disable_static(self.place)
        var_x = paddle.to_tensor(self.input_data)
        out = paddle.sort(var_x, axis=-1)
        self.assertEqual((np.sort(self.input_data, axis=-1) == out.numpy()).all(), True)
        paddle.enable_static()
if __name__ == '__main__':
    unittest.main()