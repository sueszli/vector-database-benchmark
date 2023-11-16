import unittest
import numpy as np
from op_test import OpTest
import paddle
from paddle import base
from paddle.pir_utils import test_with_pir_api

class TestTrilIndicesOp(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'tril_indices'
        self.python_api = paddle.tril_indices
        self.inputs = {}
        self.init_config()
        self.outputs = {'out': self.target}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        self.check_output(check_pir=True)

    def init_config(self):
        if False:
            print('Hello World!')
        self.attrs = {'rows': 4, 'cols': 4, 'offset': -1}
        self.target = np.tril_indices(self.attrs['rows'], self.attrs['offset'], self.attrs['cols'])
        self.target = np.array(self.target)

class TestTrilIndicesOpCase1(TestTrilIndicesOp):

    def init_config(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {'rows': 0, 'cols': 0, 'offset': 0}
        self.target = np.tril_indices(0, 0, 0)
        self.target = np.array(self.target)

class TestTrilIndicesOpCase2(TestTrilIndicesOp):

    def init_config(self):
        if False:
            while True:
                i = 10
        self.attrs = {'rows': 4, 'cols': 4, 'offset': 2}
        self.target = np.tril_indices(self.attrs['rows'], self.attrs['offset'], self.attrs['cols'])
        self.target = np.array(self.target)

class TestTrilIndicesAPICaseStatic(unittest.TestCase):

    @test_with_pir_api
    def test_static(self):
        if False:
            i = 10
            return i + 15
        places = [paddle.CPUPlace(), paddle.base.CUDAPlace(0)] if base.core.is_compiled_with_cuda() else [paddle.CPUPlace()]
        paddle.enable_static()
        for place in places:
            with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
                data1 = paddle.tril_indices(4, 4, -1)
                exe1 = paddle.static.Executor(place)
                (result1,) = exe1.run(feed={}, fetch_list=[data1])
            expected_result1 = np.tril_indices(4, -1, 4)
            np.testing.assert_allclose(result1, expected_result1, rtol=1e-05)

class TestTrilIndicesAPICaseDygraph(unittest.TestCase):

    def test_dygraph(self):
        if False:
            for i in range(10):
                print('nop')
        places = [paddle.CPUPlace(), paddle.base.CUDAPlace(0)] if base.core.is_compiled_with_cuda() else [paddle.CPUPlace()]
        for place in places:
            with base.dygraph.base.guard(place=place):
                out1 = paddle.tril_indices(4, 4, 2)
            expected_result1 = np.tril_indices(4, 2, 4)
            self.assertEqual((out1.numpy() == expected_result1).all(), True)

class TestTrilIndicesAPICaseError(unittest.TestCase):

    def test_case_error(self):
        if False:
            return 10

        def test_num_rows_type_check():
            if False:
                return 10
            out1 = paddle.tril_indices(1.0, 1, 2)
        self.assertRaises(TypeError, test_num_rows_type_check)

        def test_num_columns_type_check():
            if False:
                i = 10
                return i + 15
            out2 = paddle.tril_indices(4, -1, 2)
        self.assertRaises(TypeError, test_num_columns_type_check)

        def test_num_offset_type_check():
            if False:
                for i in range(10):
                    print('nop')
            out3 = paddle.tril_indices(4, 4, 2.0)
        self.assertRaises(TypeError, test_num_offset_type_check)

class TestTrilIndicesAPICaseDefault(unittest.TestCase):

    @test_with_pir_api
    def test_default_CPU(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            data = paddle.tril_indices(4, None, 2)
            exe = paddle.static.Executor(paddle.CPUPlace())
            (result,) = exe.run(feed={}, fetch_list=[data])
        expected_result = np.tril_indices(4, 2)
        np.testing.assert_allclose(result, expected_result, rtol=1e-05)
        with base.dygraph.base.guard(paddle.CPUPlace()):
            out = paddle.tril_indices(4, None, 2)
        expected_result = np.tril_indices(4, 2)
        self.assertEqual((out.numpy() == expected_result).all(), True)
if __name__ == '__main__':
    unittest.main()