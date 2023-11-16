from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import packed_distributed_variable
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test

class PackedDistributedVariableTest(test.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(PackedDistributedVariableTest, self).setUp()
        cpus = config.list_physical_devices('CPU')
        config.set_logical_device_configuration(cpus[0], [context.LogicalDeviceConfiguration(), context.LogicalDeviceConfiguration()])

    def testPackedVariable(self):
        if False:
            print('Hello World!')
        with ops.device('/cpu:0'):
            v0 = resource_variable_ops.ResourceVariable(1.0, name='var0')
        with ops.device('/cpu:1'):
            v1 = resource_variable_ops.ResourceVariable(2.0, name='var1')
        packed_var = packed_distributed_variable.PackedDistributedVariable([v0, v1])
        self.assertFalse(packed_var.handle.is_packed)
        self.assertTrue(packed_var.is_initialized)
        with ops.device('/cpu:0'):
            self.assertAllEqual(packed_var.get_var_on_current_device(), v0)
            val0 = packed_var.assign(2.0).assign_add(1.0)
            self.assertAllEqual(val0, 3.0)
        with ops.device('/cpu:1'):
            self.assertAllEqual(packed_var.get_var_on_current_device(), v1)
            val0 = packed_var.assign(2.0).assign_add(1.0)
            self.assertAllEqual(val0, 3.0)

        @def_function.function
        def update_var():
            if False:
                i = 10
                return i + 15
            self.assertTrue(packed_var.handle.is_packed)
            with ops.device('/cpu:0'):
                packed_var.assign_add(3.0).assign_sub(1.0)
                read0 = packed_var.value()
            with ops.device('/cpu:1'):
                packed_var.assign_sub(4.0).assign_sub(2.0)
                read1 = packed_var.value()
            return (read0, read1)
        self.assertAllEqual(update_var(), (5.0, -3.0))

    def testPackedVarAndDevice(self):
        if False:
            for i in range(10):
                print('nop')
        device0 = device_util.canonicalize('/cpu:0')
        device1 = device_util.canonicalize('/cpu:1')
        with ops.device(device0):
            v0 = resource_variable_ops.ResourceVariable(1.0)
        with ops.device(device1):
            v1 = resource_variable_ops.ResourceVariable(2.0)
        packed_var = packed_distributed_variable.PackedDistributedVariable([v0, v1])
        packed_var0 = packed_distributed_variable.PackedVarAndDevice(packed_var, device0)
        self.assertFalse(packed_var0.handle.is_packed)
        self.assertAllEqual(math_ops.mul(packed_var0, 2.0), 2.0)
        packed_var1 = packed_distributed_variable.PackedVarAndDevice(packed_var, device1)
        self.assertAllEqual(packed_var1.assign(3.0), 3.0)

        @def_function.function
        def func():
            if False:
                for i in range(10):
                    print('nop')
            self.assertTrue(packed_var.handle.is_packed)
            var0 = packed_distributed_variable.PackedVarAndDevice(packed_var, device0)
            var0.assign_add(3.0)
            var1 = packed_distributed_variable.PackedVarAndDevice(packed_var, device1)
            return (var0.value(), math_ops.add(var1, 2.0))
        self.assertAllEqual(func(), (4.0, 5.0))

    @test_util.assert_no_garbage_created
    def testNoGarbage(self):
        if False:
            return 10
        device0 = device_util.canonicalize('/cpu:0')
        device1 = device_util.canonicalize('/cpu:1')
        with ops.device(device0):
            v0 = resource_variable_ops.ResourceVariable(1.0)
        with ops.device(device1):
            v1 = resource_variable_ops.ResourceVariable(2.0)
        packed_var = packed_distributed_variable.PackedDistributedVariable([v0, v1])
        hasattr(packed_var.on_device('/cpu:0'), 'nonexist')
if __name__ == '__main__':
    ops.enable_eager_execution()
    test.main()