import os
import sys
import unittest
from site import getsitepackages
import numpy as np
from utils import check_output
import paddle
from paddle.utils.cpp_extension import load
if os.name == 'nt' or sys.platform.startswith('darwin'):
    sys.exit()
sources = ['custom_extension.cc', 'custom_sub.cc']
if paddle.is_compiled_with_cuda():
    sources.append('custom_relu_forward.cu')
paddle_includes = []
for site_packages_path in getsitepackages():
    paddle_includes.append(os.path.join(site_packages_path, 'paddle', 'include'))
    paddle_includes.append(os.path.join(site_packages_path, 'paddle', 'include', 'third_party'))
paddle_includes.append(os.path.dirname(os.path.abspath(__file__)))
custom_cpp_extension = load(name='custom_cpp_extension', sources=sources, extra_include_paths=paddle_includes, extra_cxx_cflags=['-w', '-g'], verbose=True)

class TestCppExtensionJITInstall(unittest.TestCase):
    """
    Tests setup install cpp extensions.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        SEED = 2021
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)
        self.dtypes = ['float32', 'float64']

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_cpp_extension(self):
        if False:
            while True:
                i = 10
        self._test_extension_function()
        self._test_extension_class()
        self._test_vector_tensor()
        self._test_nullable_tensor()
        self._test_optional_tensor()
        if paddle.is_compiled_with_cuda():
            self._test_cuda_relu()

    def _test_extension_function(self):
        if False:
            return 10
        for dtype in self.dtypes:
            np_x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            x = paddle.to_tensor(np_x, dtype=dtype)
            np_y = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            y = paddle.to_tensor(np_y, dtype=dtype)
            out = custom_cpp_extension.custom_add(x, y)
            target_out = np.exp(np_x) + np.exp(np_y)
            np.testing.assert_allclose(out.numpy(), target_out, atol=1e-05)
            out = custom_cpp_extension.custom_sub(x, y)
            target_out = np.exp(np_x) - np.exp(np_y)
            np.testing.assert_allclose(out.numpy(), target_out, atol=1e-05)

    def _test_extension_class(self):
        if False:
            return 10
        for dtype in self.dtypes:
            power = custom_cpp_extension.Power(3, 3)
            self.assertEqual(power.get().sum(), 9)
            self.assertEqual(power.forward().sum(), 9)
            np_x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            x = paddle.to_tensor(np_x, dtype=dtype)
            power = custom_cpp_extension.Power(x)
            np.testing.assert_allclose(power.get().sum().numpy(), np.sum(np_x), atol=1e-05)
            np.testing.assert_allclose(power.forward().sum().numpy(), np.sum(np.power(np_x, 2)), atol=1e-05)

    def _test_vector_tensor(self):
        if False:
            i = 10
            return i + 15
        for dtype in self.dtypes:
            np_inputs = [np.random.uniform(-1, 1, [4, 8]).astype(dtype) for _ in range(3)]
            inputs = [paddle.to_tensor(np_x, dtype=dtype) for np_x in np_inputs]
            out = custom_cpp_extension.custom_tensor(inputs)
            target_out = [x + 1.0 for x in inputs]
            for i in range(3):
                np.testing.assert_allclose(out[i].numpy(), target_out[i].numpy(), atol=1e-05)

    def _test_nullable_tensor(self):
        if False:
            while True:
                i = 10
        x = custom_cpp_extension.nullable_tensor(True)
        assert x is None, 'Return None when input parameter return_none = True'
        x = custom_cpp_extension.nullable_tensor(False).numpy()
        x_np = np.ones(shape=[2, 2])
        np.testing.assert_array_equal(x, x_np, err_msg=f'extension out: {x},\n numpy out: {x_np}')

    def _test_optional_tensor(self):
        if False:
            return 10
        x = custom_cpp_extension.optional_tensor(True)
        assert x is None, 'Return None when input parameter return_option = True'
        x = custom_cpp_extension.optional_tensor(False).numpy()
        x_np = np.ones(shape=[2, 2])
        np.testing.assert_array_equal(x, x_np, err_msg=f'extension out: {x},\n numpy out: {x_np}')

    def _test_cuda_relu(self):
        if False:
            print('Hello World!')
        paddle.set_device('gpu')
        x = np.random.uniform(-1, 1, [4, 8]).astype('float32')
        x = paddle.to_tensor(x, dtype='float32')
        out = custom_cpp_extension.relu_cuda_forward(x)
        pd_out = paddle.nn.functional.relu(x)
        check_output(out, pd_out, 'out')
if __name__ == '__main__':
    unittest.main()