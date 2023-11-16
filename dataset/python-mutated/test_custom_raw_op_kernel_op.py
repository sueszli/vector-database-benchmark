import importlib
import os
import shlex
import site
import sys
import unittest
import numpy as np
import paddle
MODULE_NAME = 'custom_raw_op_kernel_op_lib'

def prepare_module_path():
    if False:
        print('Hello World!')
    if os.name == 'nt':
        site_dir = site.getsitepackages()[1]
    else:
        site_dir = site.getsitepackages()[0]
    custom_egg_path = [x for x in os.listdir(site_dir) if MODULE_NAME in x]
    assert len(custom_egg_path) == 1, 'Matched egg number is %d.' % len(custom_egg_path)
    sys.path.append(os.path.join(site_dir, custom_egg_path[0]))

@unittest.skipIf(os.name == 'nt', 'Windows does not support yet.')
class TestCustomRawReluOp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(path, 'custom_raw_op_kernel_op_setup.py')
        cmd = [sys.executable, path, 'install', '--force']
        cmd = ' '.join([shlex.quote(c) for c in cmd])
        os.environ['MODULE_NAME'] = MODULE_NAME
        assert os.system(cmd) == 0
        prepare_module_path()

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        cmd = [sys.executable, '-m', 'pip', 'uninstall', '-y', MODULE_NAME]
        cmd = ' '.join([shlex.quote(c) for c in cmd])
        assert os.system(cmd) == 0

    def custom_raw_relu(self, x):
        if False:
            print('Hello World!')
        module = importlib.import_module(MODULE_NAME)
        custom_raw_relu_op = module.custom_raw_relu
        self.assertIsNotNone(custom_raw_relu_op)
        return custom_raw_relu_op(x)

    def test_static(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()
        shape = [2, 3]
        x = paddle.static.data(name='x', dtype='float32', shape=shape)
        y1 = self.custom_raw_relu(x)
        y2 = paddle.nn.ReLU()(x)
        exe = paddle.static.Executor()
        exe.run(paddle.static.default_startup_program())
        x_np = np.random.uniform(low=-1.0, high=1.0, size=[2, 3]).astype('float32')
        (y1_value, y2_value) = exe.run(paddle.static.default_main_program(), feed={x.name: x_np}, fetch_list=[y1, y2])
        np.testing.assert_array_equal(y1_value, y2_value)
        paddle.disable_static()
if __name__ == '__main__':
    unittest.main()