import os
import site
import sys
import unittest
import numpy as np
from utils import check_output, check_output_allclose
import paddle
from paddle import static
from paddle.utils.cpp_extension.extension_utils import run_cmd
from paddle.vision.transforms import Compose, Normalize

def custom_relu_dynamic(func, device, dtype, np_x, use_func=True):
    if False:
        for i in range(10):
            print('nop')
    paddle.set_device(device)
    t = paddle.to_tensor(np_x, dtype=dtype)
    t.stop_gradient = False
    t.retain_grads()
    out = func(t) if use_func else paddle.nn.functional.relu(t)
    return out.numpy()

def custom_relu_static(func, device, dtype, np_x, use_func=True, test_infer=False):
    if False:
        return 10
    paddle.enable_static()
    paddle.set_device(device)
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name='X', shape=[None, 8], dtype=dtype)
            out = func(x) if use_func else paddle.nn.functional.relu(x)
            exe = static.Executor()
            exe.run(static.default_startup_program())
            out_v = exe.run(static.default_main_program(), feed={'X': np_x}, fetch_list=[out.name])
    paddle.disable_static()
    return out_v

class TestNewCustomOpXpuSetUpInstall(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        cmd = f'cd {cur_dir} && {sys.executable} custom_relu_xpu_setup.py install'
        run_cmd(cmd)
        site_dir = site.getsitepackages()[0]
        custom_egg_path = [x for x in os.listdir(site_dir) if 'custom_relu_xpu_module_setup' in x]
        assert len(custom_egg_path) == 1, 'Matched egg number is %d.' % len(custom_egg_path)
        sys.path.append(os.path.join(site_dir, custom_egg_path[0]))
        import custom_relu_xpu_module_setup
        self.custom_op = custom_relu_xpu_module_setup.custom_relu
        self.dtypes = ['float32']
        self.device = 'xpu'
        SEED = 2021
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

    def test_static(self):
        if False:
            return 10
        for dtype in self.dtypes:
            x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            out = custom_relu_static(self.custom_op, self.device, dtype, x)
            pd_out = custom_relu_static(self.custom_op, self.device, dtype, x, False)
            check_output(out, pd_out, 'out')

    def test_dynamic(self):
        if False:
            i = 10
            return i + 15
        for dtype in self.dtypes:
            x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            out = custom_relu_dynamic(self.custom_op, self.device, dtype, x)
            pd_out = custom_relu_dynamic(self.custom_op, self.device, dtype, x, False)
            check_output(out, pd_out, 'out')

    def test_with_dataloader(self):
        if False:
            while True:
                i = 10
        paddle.disable_static()
        paddle.set_device(self.device)
        transform = Compose([Normalize(mean=[127.5], std=[127.5], data_format='CHW')])
        train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
        train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=0)
        for (batch_id, (image, _)) in enumerate(train_loader()):
            out = self.custom_op(image)
            pd_out = paddle.nn.functional.relu(image)
            check_output_allclose(out, pd_out, 'out', atol=0.01)
            if batch_id == 5:
                break
        paddle.enable_static()
if __name__ == '__main__':
    if os.name == 'nt':
        sys.exit()
    unittest.main()