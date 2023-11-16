import unittest
import numpy as np
import paddle
paddle.enable_static()

class TestOperatorBase(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.shape = [4, 16]

    def check_operator(self, operator_func, expected_out):
        if False:
            return 10
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.ones(self.shape, dtype='float32') * 2
            out = operator_func(x)
            exe = paddle.static.Executor(paddle.CPUPlace())
            res = exe.run(main_program, fetch_list=[out])
            np.testing.assert_almost_equal(res[0], expected_out)

class TestOperator(TestOperatorBase):

    def test_add(self):
        if False:
            return 10
        operator_func = lambda x: x + x
        expected_out = np.ones(self.shape, dtype='float32') * 4
        self.check_operator(operator_func, expected_out)

    def test_sub(self):
        if False:
            i = 10
            return i + 15
        operator_func = lambda x: x - x
        expected_out = np.ones(self.shape, dtype='float32') * 0
        self.check_operator(operator_func, expected_out)

    def test_mul(self):
        if False:
            return 10
        operator_func = lambda x: x * x
        expected_out = np.ones(self.shape, dtype='float32') * 4
        self.check_operator(operator_func, expected_out)

    def test_div(self):
        if False:
            for i in range(10):
                print('nop')
        operator_func = lambda x: x / x
        expected_out = np.ones(self.shape, dtype='float32') * 1
        self.check_operator(operator_func, expected_out)

class TestOperatorWithScale(TestOperatorBase):

    def test_add(self):
        if False:
            for i in range(10):
                print('nop')
        operator_func = lambda x: x + 1
        expected_out = np.ones(self.shape, dtype='float32') * 3
        self.check_operator(operator_func, expected_out)

    def test_sub(self):
        if False:
            print('Hello World!')
        operator_func = lambda x: x - 1.0
        expected_out = np.ones(self.shape, dtype='float32')
        self.check_operator(operator_func, expected_out)

    def test_mul(self):
        if False:
            i = 10
            return i + 15
        operator_func = lambda x: x * 2
        expected_out = np.ones(self.shape, dtype='float32') * 4
        self.check_operator(operator_func, expected_out)

    def test_div(self):
        if False:
            return 10
        operator_func = lambda x: x / 2.0
        expected_out = np.ones(self.shape, dtype='float32') * 1
        self.check_operator(operator_func, expected_out)

class TestCompareOperator(TestOperatorBase):

    def test_lt(self):
        if False:
            i = 10
            return i + 15
        operator_func = lambda x: x < x - 1
        expected_out = np.zeros(self.shape, dtype='bool')
        self.check_operator(operator_func, expected_out)

    def test_gt(self):
        if False:
            for i in range(10):
                print('nop')
        operator_func = lambda x: x > x - 1
        expected_out = np.ones(self.shape, dtype='bool')
        self.check_operator(operator_func, expected_out)

    def test_le(self):
        if False:
            i = 10
            return i + 15
        operator_func = lambda x: x <= x
        expected_out = np.ones(self.shape, dtype='bool')
        self.check_operator(operator_func, expected_out)

    def test_ge(self):
        if False:
            for i in range(10):
                print('nop')
        operator_func = lambda x: x >= x + 1
        expected_out = np.zeros(self.shape, dtype='bool')
        self.check_operator(operator_func, expected_out)

class TestCompareOpWithFull(TestOperatorBase):

    def test_lt(self):
        if False:
            i = 10
            return i + 15
        operator_func = lambda x: x < 1
        expected_out = np.zeros(self.shape, dtype='bool')
        self.check_operator(operator_func, expected_out)

    def test_gt(self):
        if False:
            while True:
                i = 10
        operator_func = lambda x: x > 1.0
        expected_out = np.ones(self.shape, dtype='bool')
        self.check_operator(operator_func, expected_out)

    def test_le(self):
        if False:
            while True:
                i = 10
        operator_func = lambda x: x <= 2
        expected_out = np.ones(self.shape, dtype='bool')
        self.check_operator(operator_func, expected_out)

    def test_ge(self):
        if False:
            while True:
                i = 10
        operator_func = lambda x: x >= 3.0
        expected_out = np.zeros(self.shape, dtype='bool')
        self.check_operator(operator_func, expected_out)
if __name__ == '__main__':
    unittest.main()