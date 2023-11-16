import unittest
import numpy as np
import paddle
from paddle import static
p_list_n_n = ('fro', 'nuc', 1, -1, np.inf, -np.inf)
p_list_m_n = (None, 2, -2)

def test_static_assert_true(self, x_list, p_list):
    if False:
        i = 10
        return i + 15
    for p in p_list:
        for x in x_list:
            with static.program_guard(static.Program(), static.Program()):
                input_data = static.data('X', shape=x.shape, dtype=x.dtype)
                output = paddle.linalg.cond(input_data, p)
                exe = static.Executor()
                result = exe.run(feed={'X': x}, fetch_list=[output])
                expected_output = np.linalg.cond(x, p)
                np.testing.assert_allclose(result[0], expected_output, rtol=5e-05)

def test_dygraph_assert_true(self, x_list, p_list):
    if False:
        while True:
            i = 10
    for p in p_list:
        for x in x_list:
            input_tensor = paddle.to_tensor(x)
            output = paddle.linalg.cond(input_tensor, p)
            expected_output = np.linalg.cond(x, p)
            np.testing.assert_allclose(output.numpy(), expected_output, rtol=5e-05)

def gen_input():
    if False:
        i = 10
        return i + 15
    np.random.seed(2021)
    input_1 = np.random.rand(5, 5).astype('float32')
    input_2 = np.random.rand(3, 6, 6).astype('float64')
    input_3 = np.random.rand(2, 4, 3, 3).astype('float32')
    input_4 = np.random.rand(9, 7).astype('float64')
    input_5 = np.random.rand(4, 2, 10).astype('float32')
    input_6 = np.random.rand(3, 5, 4, 1).astype('float32')
    list_n_n = (input_1, input_2, input_3)
    list_m_n = (input_4, input_5, input_6)
    return (list_n_n, list_m_n)

def gen_empty_input():
    if False:
        i = 10
        return i + 15
    input_1 = np.random.rand(0, 7, 7).astype('float32')
    input_2 = np.random.rand(0, 9, 9).astype('float32')
    input_3 = np.random.rand(0, 4, 5, 5).astype('float64')
    input_4 = np.random.rand(0, 7, 11).astype('float32')
    input_5 = np.random.rand(0, 10, 8).astype('float64')
    input_6 = np.random.rand(5, 0, 4, 3).astype('float32')
    list_n_n = (input_1, input_2, input_3)
    list_m_n = (input_4, input_5, input_6)
    return (list_n_n, list_m_n)

class API_TestStaticCond(unittest.TestCase):

    def test_out(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        (x_list_n_n, x_list_m_n) = gen_input()
        test_static_assert_true(self, x_list_n_n, p_list_n_n + p_list_m_n)
        test_static_assert_true(self, x_list_m_n, p_list_m_n)

class API_TestDygraphCond(unittest.TestCase):

    def test_out(self):
        if False:
            print('Hello World!')
        paddle.disable_static()
        (x_list_n_n, x_list_m_n) = gen_input()
        test_dygraph_assert_true(self, x_list_n_n, p_list_n_n + p_list_m_n)
        test_dygraph_assert_true(self, x_list_m_n, p_list_m_n)

class TestCondAPIError(unittest.TestCase):

    def test_dygraph_api_error(self):
        if False:
            i = 10
            return i + 15
        paddle.disable_static()
        p_list_error = ('fro_', '_nuc', -0.7, 0, 1.5, 3)
        (x_list_n_n, x_list_m_n) = gen_input()
        for p in p_list_error:
            for x in x_list_n_n + x_list_m_n:
                x_tensor = paddle.to_tensor(x)
                self.assertRaises(ValueError, paddle.linalg.cond, x_tensor, p)
        for p in p_list_n_n:
            for x in x_list_m_n:
                x_tensor = paddle.to_tensor(x)
                self.assertRaises(ValueError, paddle.linalg.cond, x_tensor, p)

    def test_static_api_error(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        p_list_error = ('f ro', 'fre', 'NUC', -1.6, 0, 5)
        (x_list_n_n, x_list_m_n) = gen_input()
        for p in p_list_error:
            for x in x_list_n_n + x_list_m_n:
                with static.program_guard(static.Program(), static.Program()):
                    x_data = static.data('X', shape=x.shape, dtype=x.dtype)
                    self.assertRaises(ValueError, paddle.linalg.cond, x_data, p)
        for p in p_list_n_n:
            for x in x_list_m_n:
                with static.program_guard(static.Program(), static.Program()):
                    x_data = static.data('X', shape=x.shape, dtype=x.dtype)
                    self.assertRaises(ValueError, paddle.linalg.cond, x_data, p)

    def test_static_empty_input_error(self):
        if False:
            print('Hello World!')
        paddle.enable_static()
        (x_list_n_n, x_list_m_n) = gen_empty_input()
        for p in p_list_n_n + p_list_m_n:
            for x in x_list_n_n:
                with static.program_guard(static.Program(), static.Program()):
                    x_data = static.data('X', shape=x.shape, dtype=x.dtype)
                    self.assertRaises(ValueError, paddle.linalg.cond, x_data, p)
        for p in p_list_n_n + p_list_m_n:
            for x in x_list_n_n:
                with static.program_guard(static.Program(), static.Program()):
                    x_data = static.data('X', shape=x.shape, dtype=x.dtype)
                    self.assertRaises(ValueError, paddle.linalg.cond, x_data, p)

class TestCondEmptyTensorInput(unittest.TestCase):

    def test_dygraph_empty_tensor_input(self):
        if False:
            while True:
                i = 10
        paddle.disable_static()
        (x_list_n_n, x_list_m_n) = gen_empty_input()
        test_dygraph_assert_true(self, x_list_n_n, p_list_n_n + p_list_m_n)
        test_dygraph_assert_true(self, x_list_m_n, p_list_m_n)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()