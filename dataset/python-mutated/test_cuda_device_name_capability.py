import unittest
import paddle

class TestDeviceName(unittest.TestCase):

    def test_device_name_default(self):
        if False:
            for i in range(10):
                print('nop')
        if paddle.is_compiled_with_cuda():
            name = paddle.device.cuda.get_device_name()
            self.assertIsNotNone(name)

    def test_device_name_int(self):
        if False:
            for i in range(10):
                print('nop')
        if paddle.is_compiled_with_cuda():
            name = paddle.device.cuda.get_device_name(0)
            self.assertIsNotNone(name)

    def test_device_name_CUDAPlace(self):
        if False:
            while True:
                i = 10
        if paddle.is_compiled_with_cuda():
            name = paddle.device.cuda.get_device_name(paddle.CUDAPlace(0))
            self.assertIsNotNone(name)

class TestDeviceCapability(unittest.TestCase):

    def test_device_capability_default(self):
        if False:
            return 10
        if paddle.is_compiled_with_cuda():
            capability = paddle.device.cuda.get_device_capability()
            self.assertIsNotNone(capability)

    def test_device_capability_int(self):
        if False:
            i = 10
            return i + 15
        if paddle.is_compiled_with_cuda():
            capability = paddle.device.cuda.get_device_capability(0)
            self.assertIsNotNone(capability)

    def test_device_capability_CUDAPlace(self):
        if False:
            while True:
                i = 10
        if paddle.is_compiled_with_cuda():
            capability = paddle.device.cuda.get_device_capability(paddle.CUDAPlace(0))
            self.assertIsNotNone(capability)
if __name__ == '__main__':
    unittest.main()