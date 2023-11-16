import unittest
import numpy as np
import paddle

class ApiOnesTest(unittest.TestCase):

    def test_paddle_ones(self):
        if False:
            print('Hello World!')
        with paddle.static.program_guard(paddle.static.Program()):
            ones = paddle.ones(shape=[10])
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[ones])
            expected_result = np.ones(10, dtype='float32')
        self.assertEqual((result == expected_result).all(), True)
        with paddle.static.program_guard(paddle.static.Program()):
            ones = paddle.ones(shape=[10], dtype='float64')
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[ones])
            expected_result = np.ones(10, dtype='float64')
        self.assertEqual((result == expected_result).all(), True)
        with paddle.static.program_guard(paddle.static.Program()):
            ones = paddle.ones(shape=[10], dtype='int64')
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[ones])
            expected_result = np.ones(10, dtype='int64')
        self.assertEqual((result == expected_result).all(), True)

    def test_base_ones(self):
        if False:
            print('Hello World!')
        with paddle.static.program_guard(paddle.static.Program()):
            ones = paddle.ones(shape=[10], dtype='int64')
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[ones])
            expected_result = np.ones(10, dtype='int64')
        self.assertEqual((result == expected_result).all(), True)

class ApiOnesZerosError(unittest.TestCase):

    def test_errors(self):
        if False:
            while True:
                i = 10

        def test_error1():
            if False:
                i = 10
                return i + 15
            with paddle.static.program_guard(paddle.static.Program()):
                ones = paddle.ones(shape=10, dtype='int64')
        self.assertRaises(TypeError, test_error1)

        def test_error2():
            if False:
                while True:
                    i = 10
            with paddle.static.program_guard(paddle.static.Program()):
                ones = paddle.ones(shape=10)
        self.assertRaises(TypeError, test_error2)

        def test_error3():
            if False:
                print('Hello World!')
            with paddle.static.program_guard(paddle.static.Program()):
                ones = paddle.ones(shape=10, dtype='int64')
        self.assertRaises(TypeError, test_error3)
if __name__ == '__main__':
    unittest.main()