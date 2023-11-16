"""Tests that the PrecisionConfig is set if TF32 is disabled."""
from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import sysconfig

class TensorFloat32Test(xla_test.XLATestCase):

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        config.enable_tensor_float_32_execution(True)

    def _test_fn(self, fn, inputs):
        if False:
            print('Hello World!')
        with ops.device('device:{}:0'.format(self.device)):
            config.enable_tensor_float_32_execution(False)
            compiled_fn = def_function.function(fn, jit_compile=True)
            hlo_text = compiled_fn.experimental_get_compiler_ir(*inputs)(stage='hlo')
            self.assertIn('operand_precision={highest,highest}', hlo_text)
            out = compiled_fn(*inputs)
            sys_details = sysconfig.get_build_info()
            if sys_details['is_rocm_build']:
                f32_out = compiled_fn(*[math_ops.cast(x, 'float32') for x in inputs])
                self.assertAllClose(out, f32_out, rtol=1e-05, atol=1e-05)
            else:
                f64_out = compiled_fn(*[math_ops.cast(x, 'float64') for x in inputs])
                self.assertAllClose(out, f64_out, rtol=1e-05, atol=1e-05)
            config.enable_tensor_float_32_execution(True)
            compiled_fn = def_function.function(fn, jit_compile=True)
            hlo_text = compiled_fn.experimental_get_compiler_ir(*inputs)(stage='hlo')
            self.assertNotIn('operand_precision', hlo_text)
            if test_util.is_gpu_available(min_cuda_compute_capability=(8, 0)):
                out = compiled_fn(*inputs)
                f64_out = compiled_fn(*[math_ops.cast(x, 'float64') for x in inputs])
                self.assertNotAllClose(out, f64_out, rtol=1e-05, atol=1e-05)

    def test_matmul(self):
        if False:
            while True:
                i = 10
        x = array_ops.fill((1024, 1024), 1 + 2 ** (-12))
        y = array_ops.fill((1024, 1024), 1.0)

        def matmul(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return math_ops.matmul(x, y)
        self._test_fn(matmul, [x, y])

    def test_batch_matmul(self):
        if False:
            while True:
                i = 10
        x = array_ops.fill((2, 1024, 1024), 1 + 2 ** (-12))
        y = array_ops.fill((2, 1024, 1024), 1.0)

        def batch_matmul(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return math_ops.matmul(x, y)
        self._test_fn(batch_matmul, [x, y])

    def test_conv2d(self):
        if False:
            print('Hello World!')
        x = array_ops.fill((16, 40, 40, 64), 1 + 2 ** (-12))
        y = array_ops.fill((3, 3, 64, 64), 1.0)

        def conv2d(x, y):
            if False:
                return 10
            return nn_ops.conv2d(x, y, [1, 1, 1, 1], padding='SAME')
        self._test_fn(conv2d, [x, y])

    def test_conv2d_backprop_input(self):
        if False:
            return 10
        y = array_ops.fill((3, 3, 64, 64), 1 + 2 ** (-12))
        out_backprop = array_ops.fill((16, 40, 40, 64), 1.0)

        def conv2d_backprop_input(y, out_backprop):
            if False:
                while True:
                    i = 10
            return nn_ops.conv2d_backprop_input((16, 40, 40, 64), y, out_backprop, [1, 1, 1, 1], padding='SAME')
        self._test_fn(conv2d_backprop_input, [y, out_backprop])

    def test_conv2d_backprop_filter(self):
        if False:
            for i in range(10):
                print('nop')
        x = array_ops.fill((16, 40, 40, 64), 1 + 2 ** (-12))
        out_backprop = array_ops.fill((16, 40, 40, 64), 1.0)

        def conv2d_backprop_filter(x, out_backprop):
            if False:
                print('Hello World!')
            return nn_ops.conv2d_backprop_filter(x, (3, 3, 64, 64), out_backprop, [1, 1, 1, 1], padding='SAME')
        self._test_fn(conv2d_backprop_filter, [x, out_backprop])
if __name__ == '__main__':
    ops.enable_eager_execution()
    config.enable_op_determinism()
    googletest.main()