from tensorflow.python.eager import context
from tensorflow.python.eager import custom_device_testutil
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import test

class CustomDeviceTest(test.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        context._reset_context()

    def testRegisterCustomDevice(self):
        if False:
            i = 10
            return i + 15
        device_name = '/job:localhost/replica:0/task:0/device:CUSTOM:0'
        (device, device_info, arrived_flag, executed_flag) = custom_device_testutil.GetLoggingDeviceCapsules(device_name)
        context.register_custom_device(device, device_name, device_info)
        self.assertFalse(custom_device_testutil.FlagValue(arrived_flag))
        self.assertFalse(custom_device_testutil.FlagValue(executed_flag))
        with ops.device(device_name):
            x = constant_op.constant(1.0)
            y = x * constant_op.constant(2.0)
        self.assertTrue(custom_device_testutil.FlagValue(executed_flag))
        self.assertFalse(custom_device_testutil.FlagValue(arrived_flag))
        with self.assertRaisesRegex(errors.InternalError, 'Trying to copy'):
            y.numpy()

    def testIsCustomDevice(self):
        if False:
            while True:
                i = 10
        device_name = '/job:localhost/replica:0/task:0/device:CUSTOM:0'
        (device, device_info, _, _) = custom_device_testutil.GetLoggingDeviceCapsules(device_name)
        context.register_custom_device(device, device_name, device_info)
        self.assertTrue(context.is_custom_device(device_name))
        self.assertFalse(context.is_custom_device('cpu:0'))
if __name__ == '__main__':
    ops.enable_eager_execution()
    test.main()