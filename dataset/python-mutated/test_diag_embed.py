import inspect
import unittest
import numpy as np
from op_test import OpTest
import paddle
from paddle import base
from paddle.base import core
from paddle.pir_utils import test_with_pir_api

class TestDiagEmbedOp(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'diag_embed'
        self.python_api = paddle.diag_embed
        self.init_config()
        self.outputs = {'Out': self.target}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_pir=True)

    def init_config(self):
        if False:
            while True:
                i = 10
        self.case = np.random.randn(2, 3).astype('float32')
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': 0, 'dim1': -2, 'dim2': -1}
        self.target = np.stack([np.diag(r, 0) for r in self.inputs['Input']], 0)

class TestDiagEmbedOpCase1(TestDiagEmbedOp):

    def init_config(self):
        if False:
            while True:
                i = 10
        self.case = np.random.randn(2, 3).astype('float32')
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': -1, 'dim1': 0, 'dim2': 2}
        self.target = np.stack([np.diag(r, -1) for r in self.inputs['Input']], 1)

class TestDiagEmbedAPICase(unittest.TestCase):

    @test_with_pir_api
    def test_case1(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            diag_embed = np.random.randn(2, 3, 4).astype('float32')
            data1 = paddle.static.data(name='data1', shape=[2, 3, 4], dtype='float32')
            out1 = paddle.diag_embed(data1)
            out2 = paddle.diag_embed(data1, offset=1, dim1=-2, dim2=3)
            place = core.CPUPlace()
            exe = base.Executor(place)
            results = exe.run(main, feed={'data1': diag_embed}, fetch_list=[out1, out2], return_numpy=True)
            target1 = np.stack([np.stack([np.diag(s, 0) for s in r], 0) for r in diag_embed], 0)
            target2 = np.stack([np.stack([np.diag(s, 1) for s in r], 0) for r in diag_embed], 0)
            np.testing.assert_allclose(results[0], target1, rtol=1e-05)
            np.testing.assert_allclose(results[1], target2, rtol=1e-05)

    def test_tensor_method(self):
        if False:
            return 10
        paddle.disable_static()
        x = paddle.arange(15).reshape((3, 5)).astype('float64')
        self.assertTrue(inspect.ismethod(x.diag_embed))
if __name__ == '__main__':
    unittest.main()