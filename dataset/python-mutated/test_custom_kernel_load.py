import os
import site
import sys
import unittest
import numpy as np

class TestCustomKernelLoad(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        cmd = 'cd {} && {} custom_kernel_dot_setup.py build_ext --inplace'.format(cur_dir, sys.executable)
        os.system(cmd)
        paddle_lib_path = ''
        site_dirs = site.getsitepackages() if hasattr(site, 'getsitepackages') else [x for x in sys.path if 'site-packages' in x]
        for site_dir in site_dirs:
            lib_dir = os.path.sep.join([site_dir, 'paddle', 'libs'])
            if os.path.exists(lib_dir):
                paddle_lib_path = lib_dir
                break
        if paddle_lib_path == '':
            if hasattr(site, 'USER_SITE'):
                lib_dir = os.path.sep.join([site.USER_SITE, 'paddle', 'libs'])
                if os.path.exists(lib_dir):
                    paddle_lib_path = lib_dir
        self.default_path = os.path.sep.join([paddle_lib_path, '..', '..', 'paddle_custom_device'])
        cmd = f'mkdir -p {self.default_path} && cp ./*.so {self.default_path}'
        os.system(cmd)

    def test_custom_kernel_dot_load(self):
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

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        cmd = f'rm -rf {self.default_path}'
        os.system(cmd)
if __name__ == '__main__':
    if os.name == 'nt' or sys.platform.startswith('darwin'):
        sys.exit()
    unittest.main()