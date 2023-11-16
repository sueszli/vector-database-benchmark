import unittest
import numpy as np
from op_test import OpTest
import paddle
from paddle import base

class TestTriuIndicesOp(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'triu_indices'
        self.python_api = paddle.triu_indices
        self.inputs = {}
        self.init_config()
        self.outputs = {'out': self.target}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        self.check_output()

    def init_config(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {'row': 4, 'col': 4, 'offset': -1}
        self.target = np.triu_indices(self.attrs['row'], self.attrs['offset'], self.attrs['col'])
        self.target = np.array(self.target)

class TestTriuIndicesOpCase1(TestTriuIndicesOp):

    def init_config(self):
        if False:
            return 10
        self.attrs = {'row': 0, 'col': 0, 'offset': 0}
        self.target = np.triu_indices(0, 0, 0)
        self.target = np.array(self.target)

class TestTriuIndicesOpCase2(TestTriuIndicesOp):

    def init_config(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {'row': 4, 'col': 4, 'offset': 2}
        self.target = np.triu_indices(self.attrs['row'], self.attrs['offset'], self.attrs['col'])
        self.target = np.array(self.target)

class TestTriuIndicesAPICaseStatic(unittest.TestCase):

    def test_static(self):
        if False:
            return 10
        if base.core.is_compiled_with_cuda():
            place = paddle.base.CUDAPlace(0)
        else:
            place = paddle.CPUPlace()
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            data = paddle.triu_indices(4, 4, -1)
            exe = paddle.static.Executor(place)
            result = exe.run(feed={}, fetch_list=[data])
        expected_result = np.triu_indices(4, -1, 4)
        np.testing.assert_array_equal(result[0], expected_result)

class TestTriuIndicesAPICaseDygraph(unittest.TestCase):

    def test_dygraph(self):
        if False:
            return 10
        if base.core.is_compiled_with_cuda():
            place = paddle.base.CUDAPlace(0)
        else:
            place = paddle.CPUPlace()
        with base.dygraph.base.guard(place=place):
            out = paddle.triu_indices(4, 4, 2)
        expected_result = np.triu_indices(4, 2, 4)
        np.testing.assert_array_equal(out, expected_result)

class TestTriuIndicesAPICaseError(unittest.TestCase):

    def test_case_error(self):
        if False:
            print('Hello World!')

        def test_num_rows_type_check():
            if False:
                return 10
            out1 = paddle.triu_indices(1.0, 1, 2)
        self.assertRaises(TypeError, test_num_rows_type_check)

        def test_num_columns_type_check():
            if False:
                for i in range(10):
                    print('nop')
            out2 = paddle.triu_indices(4, -1, 2)
        self.assertRaises(TypeError, test_num_columns_type_check)

        def test_num_offset_type_check():
            if False:
                for i in range(10):
                    print('nop')
            out3 = paddle.triu_indices(4, 4, 2.0)
        self.assertRaises(TypeError, test_num_offset_type_check)

class TestTriuIndicesAPICaseDefault(unittest.TestCase):

    def test_default_CPU(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            data = paddle.triu_indices(4, None, 2)
            exe = paddle.static.Executor(paddle.CPUPlace())
            result = exe.run(feed={}, fetch_list=[data])
        expected_result = np.triu_indices(4, 2)
        np.testing.assert_array_equal(result[0], expected_result)
        with base.dygraph.base.guard(paddle.CPUPlace()):
            out = paddle.triu_indices(4, None, 2)
        expected_result = np.triu_indices(4, 2)
        np.testing.assert_array_equal(out, expected_result)
if __name__ == '__main__':
    unittest.main()