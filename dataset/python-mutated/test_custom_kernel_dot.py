import os
import sys
import unittest
import numpy as np

class TestCustomKernelDot(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        cmd = 'cd {} && {} custom_kernel_dot_setup.py build_ext --inplace'.format(cur_dir, sys.executable)
        os.system(cmd)

    def test_custom_kernel_dot_run(self):
        if False:
            while True:
                i = 10
        x_data = np.random.uniform(1, 5, [2, 10]).astype(np.int8)
        y_data = np.random.uniform(1, 5, [2, 10]).astype(np.int8)
        result = np.sum(x_data * y_data, axis=1).reshape([2, 1])
        import paddle
        paddle.set_device('cpu')
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        out = paddle.dot(x, y)
        np.testing.assert_array_equal(out.numpy(), result, err_msg=f'custom kernel dot out: {out.numpy()},\n numpy dot out: {result}')

class TestCustomKernelDotC(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        cmd = 'cd {} && {} custom_kernel_dot_c_setup.py build_ext --inplace'.format(cur_dir, sys.executable)
        os.system(cmd)

    def test_custom_kernel_dot_run(self):
        if False:
            print('Hello World!')
        x_data = np.random.uniform(1, 5, [2, 10]).astype(np.int8)
        y_data = np.random.uniform(1, 5, [2, 10]).astype(np.int8)
        result = np.sum(x_data * y_data, axis=1).reshape([2, 1])
        import paddle
        paddle.set_device('cpu')
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        out = paddle.dot(x, y)
        np.testing.assert_array_equal(out.numpy(), result, err_msg=f'custom kernel dot out: {out.numpy()},\n numpy dot out: {result}')
if __name__ == '__main__':
    if os.name == 'nt' or sys.platform.startswith('darwin'):
        sys.exit()
    unittest.main()