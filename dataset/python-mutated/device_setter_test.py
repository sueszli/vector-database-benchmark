"""Tests for device function for replicated training."""
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import device_setter
from tensorflow.python.training import server_lib

class DeviceSetterTest(test.TestCase):
    _cluster_spec = server_lib.ClusterSpec({'ps': ['ps0:2222', 'ps1:2222'], 'worker': ['worker0:2222', 'worker1:2222', 'worker2:2222']})

    @test_util.run_deprecated_v1
    def testCPUOverride(self):
        if False:
            i = 10
            return i + 15
        with ops.device(device_setter.replica_device_setter(cluster=self._cluster_spec)):
            with ops.device('/cpu:0'):
                v = variables.Variable([1, 2])
            w = variables.Variable([2, 1])
            with ops.device('/cpu:0'):
                a = v + w
            self.assertDeviceEqual('/job:ps/task:0/cpu:0', v.device)
            self.assertDeviceEqual('/job:ps/task:0/cpu:0', v.initializer.device)
            self.assertDeviceEqual('/job:ps/task:1', w.device)
            self.assertDeviceEqual('/job:ps/task:1', w.initializer.device)
            self.assertDeviceEqual('/job:worker/cpu:0', a.device)

    @test_util.run_deprecated_v1
    def testResource(self):
        if False:
            while True:
                i = 10
        with ops.device(device_setter.replica_device_setter(cluster=self._cluster_spec)):
            v = resource_variable_ops.ResourceVariable([1, 2])
            self.assertDeviceEqual('/job:ps/task:0', v.device)

    @test_util.run_deprecated_v1
    def testPS2TasksWithClusterSpecClass(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.device(device_setter.replica_device_setter(cluster=self._cluster_spec)):
            v = variables.Variable([1, 2])
            w = variables.Variable([2, 1])
            a = v + w
            self.assertDeviceEqual('/job:ps/task:0', v.device)
            self.assertDeviceEqual('/job:ps/task:0', v.initializer.device)
            self.assertDeviceEqual('/job:ps/task:1', w.device)
            self.assertDeviceEqual('/job:ps/task:1', w.initializer.device)
            self.assertDeviceEqual('/job:worker', a.device)

    @test_util.run_deprecated_v1
    def testPS2TasksPinVariableToJob(self):
        if False:
            i = 10
            return i + 15
        with ops.device(device_setter.replica_device_setter(cluster=self._cluster_spec)):
            v = variables.Variable([1, 2])
            with ops.device('/job:moon'):
                w = variables.Variable([2, 1])
                with ops.device('/job:ps'):
                    x = variables.Variable([0, 1])
            a = v + w + x
            self.assertDeviceEqual('/job:ps/task:0', v.device)
            self.assertDeviceEqual('/job:ps/task:0', v.initializer.device)
            self.assertDeviceEqual('/job:moon', w.device)
            self.assertDeviceEqual('/job:moon', w.initializer.device)
            self.assertDeviceEqual('/job:ps/task:1', x.device)
            self.assertDeviceEqual('/job:ps/task:1', x.initializer.device)
            self.assertDeviceEqual('/job:worker', a.device)

    @test_util.run_deprecated_v1
    def testPS2TasksUseCpuForPS(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.device(device_setter.replica_device_setter(ps_tasks=1, ps_device='/cpu:0')):
            v = variables.Variable([1, 2])
            with ops.device('/job:moon'):
                w = variables.Variable([2, 1])
            a = v + w
            self.assertDeviceEqual('/cpu:0', v.device)
            self.assertDeviceEqual('/cpu:0', v.initializer.device)
            self.assertDeviceEqual('/job:moon/cpu:0', w.device)
            self.assertDeviceEqual('/job:moon/cpu:0', w.initializer.device)
            self.assertDeviceEqual('/job:worker', a.device)

    @test_util.run_deprecated_v1
    def testPS2TasksNoMerging(self):
        if False:
            while True:
                i = 10
        with ops.device(device_setter.replica_device_setter(cluster=self._cluster_spec, merge_devices=False)):
            v = variables.Variable([1, 2])
            with ops.device('/job:ps'):
                w = variables.Variable([2, 1])
            a = v + w
            self.assertDeviceEqual('/job:ps/task:0', v.device)
            self.assertDeviceEqual('/job:ps/task:0', v.initializer.device)
            self.assertDeviceEqual('/job:ps', w.device)
            self.assertDeviceEqual('/job:ps', w.initializer.device)
            self.assertDeviceEqual('/job:worker', a.device)

    @test_util.run_deprecated_v1
    def testPS2TasksWithClusterSpecDict(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.device(device_setter.replica_device_setter(cluster=self._cluster_spec.as_dict())):
            v = variables.Variable([1, 2])
            w = variables.Variable([2, 1])
            a = v + w
            self.assertDeviceEqual('/job:ps/task:0', v.device)
            self.assertDeviceEqual('/job:ps/task:0', v.initializer.device)
            self.assertDeviceEqual('/job:ps/task:1', w.device)
            self.assertDeviceEqual('/job:ps/task:1', w.initializer.device)
            self.assertDeviceEqual('/job:worker', a.device)

    @test_util.run_deprecated_v1
    def testPS2TasksWithClusterDef(self):
        if False:
            i = 10
            return i + 15
        with ops.device(device_setter.replica_device_setter(cluster=self._cluster_spec.as_cluster_def())):
            v = variables.Variable([1, 2])
            w = variables.Variable([2, 1])
            a = v + w
            self.assertDeviceEqual('/job:ps/task:0', v.device)
            self.assertDeviceEqual('/job:ps/task:0', v.initializer.device)
            self.assertDeviceEqual('/job:ps/task:1', w.device)
            self.assertDeviceEqual('/job:ps/task:1', w.initializer.device)
            self.assertDeviceEqual('/job:worker', a.device)

    @test_util.run_deprecated_v1
    def testPS2TasksWithDevice(self):
        if False:
            print('Hello World!')
        cluster_spec = server_lib.ClusterSpec({'sun': ['sun0:2222', 'sun1:2222', 'sun2:2222'], 'moon': ['moon0:2222', 'moon1:2222']})
        with ops.device(device_setter.replica_device_setter(ps_device='/job:moon', worker_device='/job:sun', cluster=cluster_spec.as_cluster_def())):
            v = variables.Variable([1, 2])
            w = variables.Variable([2, 1])
            a = v + w
            self.assertDeviceEqual('/job:moon/task:0', v.device)
            self.assertDeviceEqual('/job:moon/task:0', v.initializer.device)
            self.assertDeviceEqual('/job:moon/task:1', w.device)
            self.assertDeviceEqual('/job:moon/task:1', w.initializer.device)
            self.assertDeviceEqual('/job:sun', a.device)

    @test_util.run_deprecated_v1
    def testPS2TasksWithCPUConstraint(self):
        if False:
            print('Hello World!')
        cluster_spec = server_lib.ClusterSpec({'sun': ['sun0:2222', 'sun1:2222', 'sun2:2222'], 'moon': ['moon0:2222', 'moon1:2222']})
        with ops.device(device_setter.replica_device_setter(ps_device='/job:moon/cpu:0', worker_device='/job:sun', cluster=cluster_spec.as_cluster_def())):
            v = variables.Variable([1, 2])
            w = variables.Variable([2, 1])
            a = v + w
            self.assertDeviceEqual('/job:moon/task:0/cpu:0', v.device)
            self.assertDeviceEqual('/job:moon/task:0/cpu:0', v.initializer.device)
            self.assertDeviceEqual('/job:moon/task:1/cpu:0', w.device)
            self.assertDeviceEqual('/job:moon/task:1/cpu:0', w.initializer.device)
            self.assertDeviceEqual('/job:sun', a.device)
if __name__ == '__main__':
    test.main()