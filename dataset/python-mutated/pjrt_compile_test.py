"""Tests single device compilation + execution using the Device API (aka PjRt).

This feature is still under active development and is protected behind the
`--tf_xla_use_device_api` flag in the `TF_XLA_FLAGS` environment variable.
"""
import numpy as np
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables

class PjrtCompileTest(test.TestCase):

    def test_compile_on_demand(self):
        if False:
            for i in range(10):
                print('nop')
        if not test.is_gpu_available() or not test.is_built_with_gpu_support():
            test.skipTest('Test only applicable on GPU')
        with ops.device('/device:XLA_GPU:0'):
            a = constant_op.constant([1.0, 2.0])
            b = constant_op.constant([2.0, 3.0])
            c = a + b
            self.assertAllClose([3.0, 5.0], c, atol=1e-05)
            v = variables.Variable([0.0, 1.0])
            v.assign([1.0, 2.0])
            self.assertAllClose([1.0, 2.0], v.value(), atol=1e-05)
            v.assign_add([1.0, 2.0])
            self.assertAllClose([2.0, 4.0], v.value(), atol=1e-05)
            d = c + v
            self.assertAllClose([5.0, 9.0], d, atol=1e-05)

    def test_xla_local_launch(self):
        if False:
            print('Hello World!')
        if not test.is_gpu_available() or not test.is_built_with_gpu_support():
            test.skipTest('Test only applicable on GPU')

        @def_function.function(jit_compile=True)
        def foo(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return x + y + 1

        @def_function.function(jit_compile=True)
        def bar(x, y):
            if False:
                while True:
                    i = 10
            x.assign(y)
            y.assign_add([1.0, 1.0])
        with ops.device('/device:XLA_GPU:0'):
            self.assertEqual(self.evaluate(foo(1, 2)), 4)
            a = constant_op.constant([1.0, 2.0])
            b = constant_op.constant([2.0, 3.0])
            self.assertAllClose([4.0, 6.0], foo(a, b), atol=1e-05)
            x = variables.Variable([0.0, 1.0])
            y = variables.Variable([1.0, 2.0])
            self.assertAllClose([2.0, 4.0], foo(x, y), atol=1e-05)
            self.assertAllClose([2.0, 4.0], foo(a, x), atol=1e-05)
            bar(x, y)
            self.assertAllClose([1.0, 2.0], x.value(), atol=1e-05)
            self.assertAllClose([2.0, 3.0], y.value(), atol=1e-05)

    def test_xla_compile_and_run(self):
        if False:
            while True:
                i = 10
        pass

    def test_xla_launch_and_tf_kernel_on_gpu_device(self):
        if False:
            i = 10
            return i + 15

        @def_function.function(jit_compile=True)
        def const_fn():
            if False:
                while True:
                    i = 10
            return constant_op.constant([[1.0, 2.0], [3.0, 4.0]])

        @def_function.function(jit_compile=True)
        def matmul_fn(x):
            if False:
                while True:
                    i = 10
            return math_ops.matmul(x, x)
        host_tensor = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
        with ops.device('/device:GPU:0'):
            xla_tensor = const_fn()
            xla_result = matmul_fn(host_tensor)
            result = math_ops.matmul(xla_result, xla_tensor)
        ref_tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
        ref_result = np.matmul(np.matmul(ref_tensor, ref_tensor), ref_tensor)
        self.assertAllClose(result.numpy(), ref_result, atol=1e-05)
        with ops.device('/device:GPU:0'):
            tf_matmul_tensor = math_ops.matmul(host_tensor, host_tensor)
            xla_result = matmul_fn(tf_matmul_tensor)
        ref_matmul_tensor = np.matmul(ref_tensor, ref_tensor)
        ref_result_2 = np.matmul(ref_matmul_tensor, ref_matmul_tensor)
        self.assertAllClose(xla_result.numpy(), ref_result_2, atol=1e-05)

    def test_xla_launch_with_var_on_gpu_device(self):
        if False:
            while True:
                i = 10

        @def_function.function(jit_compile=True)
        def foo(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return x + y + 1

        @def_function.function(jit_compile=True)
        def bar(x, y):
            if False:
                i = 10
                return i + 15
            x.assign(y)
            y.assign_add([1.0, 1.0])
        with ops.device('/device:GPU:0'):
            a = constant_op.constant([1.0, 2.0])
            x = variables.Variable([0.0, 1.0])
            result_tensor = foo(x, a)
        self.assertAllClose(result_tensor.numpy(), [2.0, 4.0], atol=1e-05)
        with ops.device('/device:GPU:0'):
            var_a = variables.Variable([0.0, 1.0])
            var_b = variables.Variable([1.0, 2.0])
            bar(var_a, var_b)
            result = foo(var_a, var_b)
        self.assertAllClose([1.0, 2.0], var_a.value(), atol=1e-05)
        self.assertAllClose([2.0, 3.0], var_b.value(), atol=1e-05)
        self.assertAllClose(result, [4.0, 6.0], atol=1e-05)
if __name__ == '__main__':
    test.main()