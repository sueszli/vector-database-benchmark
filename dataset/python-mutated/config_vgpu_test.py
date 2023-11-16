"""Tests that the system configuration methods work properly."""
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.platform import test

def reset_eager(fn):
    if False:
        return 10

    def wrapper(*args, **kwargs):
        if False:
            print('Hello World!')
        try:
            return fn(*args, **kwargs)
        finally:
            context._reset_jit_compiler_flags()
            context._reset_context()
            ops.enable_eager_execution_internal()
            assert context._context is not None
    return wrapper

class DeviceTest(test.TestCase):
    already_run = False

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        if self.already_run:
            raise RuntimeError('Each test in this test suite must run in a separate process. Increase number of shards used to run this test.')
        self.already_run = True

    @reset_eager
    def testVGpu1P2V(self):
        if False:
            return 10
        gpus = config.list_physical_devices('GPU')
        if len(gpus) != 1:
            self.skipTest('Need 1 GPUs')
        config.set_logical_device_configuration(gpus[0], [context.LogicalDeviceConfiguration(memory_limit=100, experimental_device_ordinal=0), context.LogicalDeviceConfiguration(memory_limit=100, experimental_device_ordinal=1)])
        context.ensure_initialized()
        vcpus = config.list_logical_devices('GPU')
        self.assertEqual(len(vcpus), 2)

    @reset_eager
    def testVGpu2P2V1Default(self):
        if False:
            print('Hello World!')
        gpus = config.list_physical_devices('GPU')
        if len(gpus) != 2:
            self.skipTest('Need 2 GPUs')
        config.set_logical_device_configuration(gpus[0], [context.LogicalDeviceConfiguration(memory_limit=100), context.LogicalDeviceConfiguration(memory_limit=100)])
        context.ensure_initialized()
        vcpus = config.list_logical_devices('GPU')
        self.assertEqual(len(vcpus), 3)

    @reset_eager
    def testGpu2P2V2V(self):
        if False:
            i = 10
            return i + 15
        gpus = config.list_physical_devices('GPU')
        if len(gpus) != 2:
            self.skipTest('Need 2 GPUs')
        config.set_logical_device_configuration(gpus[0], [context.LogicalDeviceConfiguration(memory_limit=100, experimental_device_ordinal=0), context.LogicalDeviceConfiguration(memory_limit=100, experimental_device_ordinal=1)])
        config.set_logical_device_configuration(gpus[1], [context.LogicalDeviceConfiguration(memory_limit=100, experimental_device_ordinal=0), context.LogicalDeviceConfiguration(memory_limit=100, experimental_device_ordinal=1)])
        context.ensure_initialized()
        vcpus = config.list_logical_devices('GPU')
        self.assertEqual(len(vcpus), 4)

    @reset_eager
    def testGpu2P2D2D(self):
        if False:
            while True:
                i = 10
        gpus = config.list_physical_devices('GPU')
        if len(gpus) != 2:
            self.skipTest('Need 2 GPUs')
        config.set_logical_device_configuration(gpus[0], [context.LogicalDeviceConfiguration(memory_limit=100), context.LogicalDeviceConfiguration(memory_limit=100)])
        config.set_logical_device_configuration(gpus[1], [context.LogicalDeviceConfiguration(memory_limit=100), context.LogicalDeviceConfiguration(memory_limit=100)])
        context.ensure_initialized()
        vcpus = config.list_logical_devices('GPU')
        self.assertEqual(len(vcpus), 4)
if __name__ == '__main__':
    ops.enable_eager_execution()
    test.main()