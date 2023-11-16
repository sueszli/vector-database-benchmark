"""Tests virtual device compilation + execution using the Device API (aka PjRt).

This feature is still under active development and is protected behind the
`--tf_xla_use_device_api` flag in the `TF_XLA_FLAGS` environment variable.
"""
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables

class PjrtCompileVirtualDeviceTest(test.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        gpus = config.list_physical_devices('GPU')
        config.set_logical_device_configuration(gpus[0], [context.LogicalDeviceConfiguration(memory_limit=1024), context.LogicalDeviceConfiguration(memory_limit=1024), context.LogicalDeviceConfiguration(memory_limit=1024)])

    def test_xla_launch_and_tf_kernel_on_gpu_device(self):
        if False:
            i = 10
            return i + 15

        @def_function.function(jit_compile=True)
        def foo(x, y):
            if False:
                return 10
            return x + y + 1

        @def_function.function(jit_compile=True)
        def bar(x, y):
            if False:
                for i in range(10):
                    print('nop')
            x.assign(y)
            y.assign_add([1.0, 1.0])
        with ops.device('/device:GPU:1'):
            a = constant_op.constant([1.0, 2.0])
            x = variables.Variable([0.0, 1.0])
            result_tensor = foo(x, a)
        self.assertAllClose(result_tensor.numpy(), [2.0, 4.0], atol=1e-05)
        with ops.device('/device:GPU:1'):
            var_a = variables.Variable([0.0, 1.0])
            var_b = variables.Variable([1.0, 2.0])
            bar(var_a, var_b)
            result = foo(var_a, var_b)
        self.assertAllClose([1.0, 2.0], var_a.value(), atol=1e-05)
        self.assertAllClose([2.0, 3.0], var_b.value(), atol=1e-05)
        self.assertAllClose(result, [4.0, 6.0], atol=1e-05)
if __name__ == '__main__':
    test.main()