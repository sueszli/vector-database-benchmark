import os
import site
import sys
import unittest
import numpy as np
import paddle
from paddle import static
from paddle.utils.cpp_extension.extension_utils import run_cmd

def custom_relu_static(func, device, dtype, np_x, use_func=True, test_infer=False):
    if False:
        return 10
    paddle.enable_static()
    paddle.set_device(device)
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name='X', shape=[None, 8], dtype=dtype)
            x.stop_gradient = False
            out = func(x) if use_func else paddle.nn.functional.relu(x)
            static.append_backward(out)
            exe = static.Executor()
            exe.run(static.default_startup_program())
            out_v = exe.run(static.default_main_program(), feed={'X': np_x}, fetch_list=[out.name])
    paddle.disable_static()
    return out_v

def custom_relu_dynamic(func, device, dtype, np_x, use_func=True):
    if False:
        return 10
    paddle.set_device(device)
    t = paddle.to_tensor(np_x, dtype=dtype)
    t.stop_gradient = False
    out = func(t) if use_func else paddle.nn.functional.relu(t)
    out.stop_gradient = False
    out.backward()
    if t.grad is None:
        return (out.numpy(), t.grad)
    else:
        return (out.numpy(), t.grad.numpy())

def custom_relu_double_grad_dynamic(func, device, dtype, np_x, use_func=True):
    if False:
        while True:
            i = 10
    paddle.set_device(device)
    t = paddle.to_tensor(np_x, dtype=dtype, stop_gradient=False)
    t.retain_grads()
    out = func(t) if use_func else paddle.nn.functional.relu(t)
    out.retain_grads()
    dx = paddle.grad(outputs=out, inputs=t, grad_outputs=paddle.ones_like(t), create_graph=True, retain_graph=True)
    ddout = paddle.grad(outputs=dx[0], inputs=out.grad, grad_outputs=paddle.ones_like(t), create_graph=False)
    assert ddout[0].numpy() is not None
    return (dx[0].numpy(), ddout[0].numpy())

class TestCppExtensionSetupInstall(unittest.TestCase):
    """
    Tests setup install cpp extensions.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        cmd = 'cd {} && {} mix_relu_and_extension_setup.py install'.format(cur_dir, sys.executable)
        run_cmd(cmd)
        site_dir = site.getsitepackages()[0]
        custom_egg_path = [x for x in os.listdir(site_dir) if 'mix_relu_extension' in x]
        assert len(custom_egg_path) == 1, 'Matched egg number is %d.' % len(custom_egg_path)
        sys.path.append(os.path.join(site_dir, custom_egg_path[0]))
        SEED = 2021
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)
        self.dtypes = ['float32', 'float64']

    def tearDown(self):
        if False:
            return 10
        pass

    def test_cpp_extension(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_extension_function_mixed()
        self._test_static()
        self._test_dynamic()
        self._test_double_grad_dynamic()

    def _test_extension_function_mixed(self):
        if False:
            return 10
        import mix_relu_extension
        for dtype in self.dtypes:
            np_x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            x = paddle.to_tensor(np_x, dtype=dtype)
            np_y = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            y = paddle.to_tensor(np_y, dtype=dtype)
            out = mix_relu_extension.custom_add2(x, y)
            target_out = np.exp(np_x) + np.exp(np_y)
            np.testing.assert_allclose(out.numpy(), target_out, atol=1e-05)
            out = mix_relu_extension.custom_sub2(x, y)
            target_out = np.exp(np_x) - np.exp(np_y)
            np.testing.assert_allclose(out.numpy(), target_out, atol=1e-05)

    def _test_static(self):
        if False:
            return 10
        import mix_relu_extension
        for dtype in self.dtypes:
            x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            out = custom_relu_static(mix_relu_extension.custom_relu, 'CPU', dtype, x)
            pd_out = custom_relu_static(mix_relu_extension.custom_relu, 'CPU', dtype, x, False)
            np.testing.assert_array_equal(out, pd_out, err_msg=f'custom op out: {out},\n paddle api out: {pd_out}')

    def _test_dynamic(self):
        if False:
            print('Hello World!')
        import mix_relu_extension
        for dtype in self.dtypes:
            x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            (out, x_grad) = custom_relu_dynamic(mix_relu_extension.custom_relu, 'CPU', dtype, x)
            (pd_out, pd_x_grad) = custom_relu_dynamic(mix_relu_extension.custom_relu, 'CPU', dtype, x, False)
            np.testing.assert_array_equal(out, pd_out, err_msg=f'custom op out: {out},\n paddle api out: {pd_out}')
            np.testing.assert_array_equal(x_grad, pd_x_grad, err_msg=f'custom op x grad: {x_grad},\n paddle api x grad: {pd_x_grad}')

    def _test_double_grad_dynamic(self):
        if False:
            i = 10
            return i + 15
        import mix_relu_extension
        for dtype in self.dtypes:
            x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            (out, dx_grad) = custom_relu_double_grad_dynamic(mix_relu_extension.custom_relu, 'CPU', dtype, x)
            (pd_out, pd_dx_grad) = custom_relu_double_grad_dynamic(mix_relu_extension.custom_relu, 'CPU', dtype, x, False)
            np.testing.assert_array_equal(out, pd_out, err_msg=f'custom op out: {out},\n paddle api out: {pd_out}')
            np.testing.assert_array_equal(dx_grad, pd_dx_grad, err_msg='custom op dx grad: {},\n paddle api dx grad: {}'.format(dx_grad, pd_dx_grad))
if __name__ == '__main__':
    if os.name == 'nt' or sys.platform.startswith('darwin'):
        sys.exit()
    unittest.main()