"""Tests for the SWIG-wrapped device lib."""
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import device_lib
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test

class DeviceLibTest(test_util.TensorFlowTestCase):

    def testListLocalDevices(self):
        if False:
            i = 10
            return i + 15
        devices = device_lib.list_local_devices()
        self.assertGreater(len(devices), 0)
        self.assertEqual(devices[0].device_type, 'CPU')
        devices = device_lib.list_local_devices(config_pb2.ConfigProto())
        self.assertGreater(len(devices), 0)
        self.assertEqual(devices[0].device_type, 'CPU')
        if test.is_gpu_available():
            self.assertGreater(len(devices), 1)
            self.assertIn('GPU', [d.device_type for d in devices])
if __name__ == '__main__':
    googletest.main()