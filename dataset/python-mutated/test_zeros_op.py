import unittest
import numpy as np
import paddle
from paddle import base
from paddle.base import Program, program_guard

class ApiZerosTest(unittest.TestCase):

    def test_out(self):
        if False:
            return 10
        with program_guard(Program()):
            zeros = paddle.zeros(shape=[10], dtype='float64')
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[zeros])
            expected_result = np.zeros(10, dtype='float64')
        self.assertEqual((result == expected_result).all(), True)
        with paddle.static.program_guard(Program()):
            zeros = paddle.zeros(shape=[10], dtype='int64')
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[zeros])
            expected_result = np.zeros(10, dtype='int64')
        self.assertEqual((result == expected_result).all(), True)
        with program_guard(Program()):
            zeros = paddle.zeros(shape=[10], dtype='int8')
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[zeros])
            expected_result = np.zeros(10, dtype='int8')
        self.assertEqual((result == expected_result).all(), True)
        with program_guard(Program()):
            out_np = np.zeros(shape=1, dtype='float32')
            out = paddle.zeros(shape=[1], dtype='float32')
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            result = exe.run(fetch_list=[out])
            self.assertEqual((result == out_np).all(), True)

    def test_base_out(self):
        if False:
            i = 10
            return i + 15
        with program_guard(Program()):
            zeros = paddle.zeros(shape=[10], dtype='int64')
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[zeros])
            expected_result = np.zeros(10, dtype='int64')
        self.assertEqual((result == expected_result).all(), True)

class ApiZerosError(unittest.TestCase):

    def test_errors(self):
        if False:
            for i in range(10):
                print('nop')

        def test_error1():
            if False:
                while True:
                    i = 10
            with paddle.static.program_guard(base.Program()):
                ones = paddle.zeros(shape=10, dtype='int64')
        self.assertRaises(TypeError, test_error1)

    def test_shape_errors(self):
        if False:
            print('Hello World!')
        with base.dygraph.guard():
            try:
                shape = [-1, 5]
                out = paddle.zeros(shape)
            except Exception as e:
                error_msg = str(e)
                assert error_msg.find('expected to be no less than 0') > 0
if __name__ == '__main__':
    unittest.main()