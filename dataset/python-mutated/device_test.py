"""Tests for tensorflow.python.framework.device."""
from absl.testing import parameterized
from tensorflow.python.eager import context
from tensorflow.python.framework import device
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
TEST_V1_AND_V2 = (('v1', device_spec.DeviceSpecV1), ('v2', device_spec.DeviceSpecV2))

class DeviceTest(test_util.TensorFlowTestCase, parameterized.TestCase):

    @parameterized.named_parameters(*TEST_V1_AND_V2)
    def testMerge(self, DeviceSpec):
        if False:
            while True:
                i = 10
        d = DeviceSpec.from_string('/job:muu/task:1/device:MyFunnyDevice:2')
        self.assertEqual('/job:muu/task:1/device:MyFunnyDevice:2', d.to_string())
        if not context.executing_eagerly():
            with ops.device(device.merge_device('/device:GPU:0')):
                var1 = variables.Variable(1.0)
                self.assertEqual('/device:GPU:0', var1.device)
                with ops.device(device.merge_device('/job:worker')):
                    var2 = variables.Variable(1.0)
                    self.assertEqual('/job:worker/device:GPU:0', var2.device)
                    with ops.device(device.merge_device('/device:CPU:0')):
                        var3 = variables.Variable(1.0)
                        self.assertEqual('/job:worker/device:CPU:0', var3.device)
                        with ops.device(device.merge_device('/job:ps')):
                            var4 = variables.Variable(1.0)
                            self.assertEqual('/job:ps/device:CPU:0', var4.device)

    def testCanonicalName(self):
        if False:
            return 10
        self.assertEqual('/job:foo/replica:0', device.canonical_name('/job:foo/replica:0'))
        self.assertEqual('/job:foo/replica:0', device.canonical_name('/replica:0/job:foo'))
        self.assertEqual('/job:foo/replica:0/task:0', device.canonical_name('/job:foo/replica:0/task:0'))
        self.assertEqual('/job:foo/replica:0/task:0', device.canonical_name('/job:foo/task:0/replica:0'))
        self.assertEqual('/device:CPU:0', device.canonical_name('/device:CPU:0'))
        self.assertEqual('/device:GPU:2', device.canonical_name('/device:GPU:2'))
        self.assertEqual('/job:foo/replica:0/task:0/device:GPU:0', device.canonical_name('/job:foo/replica:0/task:0/device:GPU:0'))
        self.assertEqual('/job:foo/replica:0/task:0/device:GPU:0', device.canonical_name('/device:GPU:0/task:0/replica:0/job:foo'))

    def testCheckValid(self):
        if False:
            i = 10
            return i + 15
        device.check_valid('/job:foo/replica:0')
        with self.assertRaisesRegex(ValueError, 'invalid literal for int'):
            device.check_valid('/job:j/replica:foo')
        with self.assertRaisesRegex(ValueError, 'invalid literal for int'):
            device.check_valid('/job:j/task:bar')
        with self.assertRaisesRegex(ValueError, "Unknown attribute 'barcpugpu'"):
            device.check_valid('/barcpugpu:muu/baz:2')
        with self.assertRaisesRegex(ValueError, 'Multiple device types are not allowed'):
            device.check_valid('/cpu:0/device:GPU:2')
if __name__ == '__main__':
    googletest.main()