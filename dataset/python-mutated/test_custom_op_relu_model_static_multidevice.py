import os
import subprocess
import tempfile
import unittest

class TestCustomOpReluModelStaticMultiDevice(unittest.TestCase):

    def install_custom_op(self):
        if False:
            return 10
        cmds = ['python', 'setup_for_static_multidevice_test.py', 'install']
        p = subprocess.run(cmds)
        assert p.returncode == 0, f'Install Custom Op: Failed: {p}'

    def setUp(self):
        if False:
            print('Hello World!')
        self.fleet_log_dir = tempfile.TemporaryDirectory()
        self.model_dir = tempfile.TemporaryDirectory()
        self.output_log_dir = tempfile.TemporaryDirectory()
        self.install_custom_op()

    def train(self, use_custom_op: bool=True):
        if False:
            for i in range(10):
                print('nop')
        cmds = ['python', '-m', 'paddle.distributed.launch']
        cmds += ['--log_dir', self.fleet_log_dir.name]
        cmds += ['custom_op_multidevice_model_train.py']
        cmds += ['--output_dir', self.output_log_dir.name]
        cmds += ['--model_dir', self.model_dir.name]
        if use_custom_op:
            cmds += ['--use_custom_op']
        cmds += ['--train_mode']
        p = subprocess.run(cmds)
        assert p.returncode == 0, f'Fleet train: Failed: {p}'

    def eval(self, use_custom_op: bool=True):
        if False:
            for i in range(10):
                print('nop')
        cmds = ['python', '-m', 'paddle.distributed.launch']
        cmds += ['--log_dir', self.fleet_log_dir.name]
        cmds += ['custom_op_multidevice_model_train.py']
        cmds += ['--output_dir', self.output_log_dir.name]
        cmds += ['--model_dir', self.model_dir.name]
        if use_custom_op:
            cmds += ['--use_custom_op']
        p = subprocess.run(cmds)
        assert p.returncode == 0, f'Fleet eval: Failed: {p}'

    def tearDown(self):
        if False:
            print('Hello World!')
        self.fleet_log_dir.cleanup()
        self.model_dir.cleanup()
        self.output_log_dir.cleanup()

    def test_train_and_eval(self):
        if False:
            print('Hello World!')
        self.train(use_custom_op=True)
        self.train(use_custom_op=False)
        import numpy as np
        import paddle
        count = 0
        if paddle.framework.core.is_compiled_with_cuda():
            count = paddle.framework.core.get_cuda_device_count()
        elif paddle.framework.core.is_compiled_with_xpu():
            count = paddle.framework.core.get_xpu_device_count()
        assert count > 1, 'TestCustomOpReluModelStaticMultiDevice needs at least two devices'
        for id in range(count):
            loss_custom = np.load(os.path.join(self.output_log_dir.name, f'train_{id}_{True}.npz'))
            loss_origin = np.load(os.path.join(self.output_log_dir.name, f'train_{id}_{False}.npz'))
            np.testing.assert_array_equal(loss_custom['losses'], loss_origin['losses'])
            np.testing.assert_array_equal(loss_custom['relu_out1_list'], loss_origin['relu_out1_list'])
            np.testing.assert_array_equal(loss_custom['relu_out2_list'], loss_origin['relu_out2_list'])
        self.eval(use_custom_op=True)
        self.eval(use_custom_op=False)
        for id in range(count):
            loss_custom = np.load(os.path.join(self.output_log_dir.name, f'eval_{id}_{True}.npz'))
            loss_origin = np.load(os.path.join(self.output_log_dir.name, f'eval_{id}_{False}.npz'))
            np.testing.assert_array_equal(loss_custom['losses'], loss_origin['losses'])
            np.testing.assert_array_equal(loss_custom['relu_out1_list'], loss_origin['relu_out1_list'])
            np.testing.assert_array_equal(loss_custom['relu_out2_list'], loss_origin['relu_out2_list'])
if __name__ == '__main__':
    unittest.main()